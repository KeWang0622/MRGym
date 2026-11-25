# Why thinking-off Qwen3.8-27B fails SNRSpec, FatSuppress, ArtefactClinic (run c04) — 2026-11-25

Read-only. Episode paths: under `runs/c04/records/cal/episodes/qwen3.8-27b/Scanner-<Family>-v0/`.
Code paths: under `src/mrgym/`. Scripts and intermediate files:
`analyst-scratch/sfc/` (the parent folder is shared; `table.py`, `dump.py`, `errs.py` there are not mine).

## 0. Most valuable finding: ArtefactClinic instances are not a function of (seed, level) under the threaded runner
- Fact (stored data): in 5 of 12 c04 cells (L1 s9000, s9003; L2 s9000, s9002, s9003) the task text the model saw differs
  from the text this checkout builds from (level, seed): other finding positions, other protocol, other requirement.
  L1/s9000 holds three different instances under one `task_id` (e0: budget 61 s; e2; e1, e3–e7: budget 65 s).
- Reproduced (`sfc/race_test4.py`, one process, 2 threads): building L1 seed 9000 alone or in either serial order tested gives
  attempt 2, budget 65 s; building it while another thread builds the L3 instance gives attempt 4, budget 61 s, and its
  card hash `c2985d` equals the stored c04 variant of L1/s9000/e0. Three threads on L1 only: no change.
  `agents/runner.py` runs model episodes in a `ThreadPoolExecutor`; `artefact_clinic/task.py::_instance` is an
  `lru_cache` that does not serialise concurrent builds.
- Root cause not located. Suspects, unverified: shared cached arrays in `scanner/rf.py` (lru_cache at lines 80, 105, 112)
  or `scanner/epg.py:110`, used by the L3 constructed build.
- Consequences: `mrgym verify` / any rebuild from the seed scores another instance for these 40 episodes; groups keyed
  by `task_id` mix different tasks; the bank docstring ("deterministic in (seed, level)") does not hold under threads.
- Scoring itself checked out: serial re-scoring reproduces every stored check on the cells that match (ArtefactClinic
  56/56 episodes, SNRSpec 162/162 image checks, FatSuppress 475/475 region checks; `sfc/recheck.py`). All 24 SNRSpec and
  FatSuppress cells rebuild to the stored text.

## 1. Table: first failing check (declared order), all 288 records
| Family L | pass | outcome | first failing check | any failing | `distance.error` |
|---|---|---|---|---|---|
| SNRSpec 1 | 0/32 | failed 30, no submission 2 | snr_A 16, time_A 13, snr_B 1 | snr_B 21, time_A 21, time_B 20, snr_A 16 | summed delivered scan s: 19 / 350 / 2938 (min/med/max) |
| SNRSpec 2 | 0/32 | failed 27, no submission 5 | snr_A 26, time_A 1 | snr_B 27, snr_A 26, time_A 4, time_B 4 | 20 / 478 / 18440 |
| SNRSpec 3 | 0/32 | failed 24, no submission 8 | snr_A 19, time_A 5 | snr_B 23, snr_A 19, time_A 6, time_B 1 | 59 / 518 / 1399 |
| FatSuppress 1 | 0/32 | failed 32 | suppressed_* 32 | suppressed 32, method 10 | worst residual: 0.17 / 1.00 / 1.00 (limit 0.05) |
| FatSuppress 2 | 1/32 | failed 31 | suppressed_* 31 | suppressed 31, method 3 | 0.047 (the pass) / 0.94 / 1.00 |
| FatSuppress 3 | 0/32 | failed 31, no submission 1 | suppressed_* 30, method_* 1 | suppressed 31, method 7 | 0.16 / 0.70 / 1.00 |
| Clinic 1 | 0/32 | failed 32 | diagnosis_* 32 | diagnosis 32, clean 27, locked 11, scan_time 3, snr 3 | worst E / threshold: 0.54 / 4.4 / 73 |
| Clinic 2 | 0/32 | failed 31, no submission 1 | diagnosis_* 28, snr 1, locked 1, clean 1 | diagnosis 28, clean 27, snr 5, scan_time 5, locked 3 | 0.36 / 4.1 / 40 |
| Clinic 3 | 0/32 | failed 32 | diagnosis_* 32 | diagnosis 32, clean 31, locked 5, scan_time 4, snr 2, te_window 2 | 0.21 / 13.2 / 20 |

SNRSpec `distance.error` is a cost, not a distance to the tolerance. Re-scored against the hidden optimum
(`sfc/snr_rescore.txt`), per delivered image: L1 pass 1, SNR low 37, too long 22 (median SNR margin −10 %, time 1.66 ×);
L2 SNR low 53 of 54 (median −88 %); L3 SNR low 42 of 48 (median −68 %). Clinic findings diagnosed right per episode
(stored checks): L1 {0: 8, 1: 11, 2: 9, 3: 4}, L2 {0: 11, 1: 9, 2: 6, 3: 2, 4: 3}, L3 {0: 8, 1: 17, 2: 5, 3: 2}.

## 2. SNRSpec (0/96)
Pass needs, per image, SNR ≥ requirement and scan time ≤ 1.15 × the hidden optimum (`snr_spec/task.py::score`), for
both images. On averages alone 15 % of time is 7 % of SNR: the prediction has to be right to a few percent.
Reference (`snr_spec/controls.py::_measure`): wide-FOV noise scan, matrix-64 scouts at 4 TR and 2 TE, fit, 3-D search.

**1. Classes, 13 episodes read** — (b) 8, (a) 3, (c) 2.
- (b) wrong method: L1/s9000/e0 (T1 assumed 100 ms from one TR, BW held at 300 as "minimum"), L1/s9001/e5 (312 of 330 s
  on two full-matrix TR-1000 scans, T1 never measured), L1/s9002/e3 (signal modelled as proportional to scan time),
  L2/s9000/e5 (SNR taken to rise with bandwidth, delivers 3000 Hz/px), L2/s9001/e5 (247 s on two 2-average full-matrix
  scans, then a recovery formula that cannot be fitted), L3/s9000/e3 ("TR doesn't help", one TR for image B),
  L3/s9002/e3 (concludes "averaging does not help", the signal is "a random phasing artifact"), L3/s9003/e2 (TR 50 ms
  with 28 averages: SNR 3.58 against 50.8).
- (a) interface wall: L1/s9002/e1, L2/s9002/e2, L3/s9001/e6 — ended at the rejected-call limit, mostly `calc` refusals.
  Their plans were also wrong (e.g. L2/s9002/e2), so removing the wall would not have passed them.
- (c) right ingredients, sloppy: L1/s9003/e6 (T2 = 8.8 ms fitted from noise-level TR-50 readings; scan time doubled:
  "TA = 3 + 96*500/1000*(1+1) = 99s"), L2/s9003/e5 (3 TR, 4 BW, background noise, then σ = sd|S|/√2 and "σ linear in BW").
- Snippets of the dominant failure:
  L2/s9002/e2: "Since SNR ∝ √av (not linear), averages never help vs TR, and noise ∝ √BW (favor lowest BW=300)."
  L2/s9000/e5: "# SNR = |mean|/sd * sqrt(N) * sqrt(bw/300)" … "ALL configs over budget - need to recheck noise model".
- Whole-run counts behind this (`sfc/feat_snr.py`, `sfc/snr_cls.py`): 61 of 96 episodes scanned ≥ 3 TR, yet delivered
  bandwidth is 50 Hz/px in 36 of 162 images, 300 in 82, above 300 in 29; TR = 50 ms (console minimum) in 43 of 162.

**2. Statement vs verifier.** Both checks are stated and match `score`. Two things the text leaves open:
- Budget vs delivered scan. At L2 the optimum is 263–2686 s against a 330 s probing budget. Where the optimum exceeds
  330 s the model delivered a scan ≤ 330 s in 29 of 48 images (L2) and 17 of 31 (L3). L1/s9003/e6: "Total: 99+227 = 326s
  <= 330s ✓". The text says the images "are not run … and are not charged" but never that they may exceed the budget.
- Noise depends on the matrix (`controls.py::noise_factor`); the datasheet only says it "follows from the
  reconstruction". Not a defect, but most episodes carry one noise number across matrices.
- Near misses by hand: L1/s9002/e3 image A TR 800, 1 avg, 50 Hz/px = 80.6 s (0.99 × optimum 81.55 s), SNR 49.8 against
  50.3 (−1 %), needed TR ≥ 809.8; image B 67.4 s against 28.9 s (2.33 ×). L3/s9003/e2 image B SNR 11.6 against 12.1 (−4 %)
  at 0.96 ×; image A SNR 3.58 against 50.8. L1/s9001/e5 both images SNR met (+33 %, +22 %) but 1.36 × and 1.27 ×.
  The L1/s9002/e3 hit is luck: its model was "signal proportional to scan time".
**3. Distance.** Nowhere near. One image of 162 passes (L1/s9003/e6 image A), three more are within 2 × the tolerance.

**4. One change.** None converts failures: the gap is competence. Two changes remove noise around it: say in the
statement that a delivered scan may be much longer than the budget; stop counting `calc` refusals towards the 8
rejected calls and state the step cap in the tool description.

**5. Practice level / hint tier.** One image, one average, bandwidth given (so only TR is free), plus hints: noise =
mean|S| of an empty corner / 1.253 on a short-TR scan at the image's own matrix and bandwidth; signal of each region
at ≥ 3 TR on matrix-64 scouts, fit A·(1 − e^(−TR/T1)); SNR scales with √(averages / bandwidth). Not tested here.

**6. Defects (interface, not verifier).**
- `calc` stops at 200 000 steps: hit in 46 of 96 episodes, 130 refusals. The tool description does not state the cap.
  Refusals count towards the episode-ending limit while the status line says "rejected 5/8 (nothing charged)". 14 of the
  15 no-submission episodes ended at that limit; 91 of their 112 rejected calls are `calc`.
- `plan` prints "minimums: te_ms 8.4, tr_ms 12.14" for `se`, whose range is 50–20000: 41 of 96 episodes then send a
  TR below 50 and get "tr_ms: 12.14 is outside 50 to 20000 for se" (L3/s9003/e2 even submits 12.12).
- Run c01 says little about competence here: 74 of 96 episodes stopped with `no_tool_calls`, 15 submitted, 1 passed.

## 3. FatSuppress (1/96)
Pass needs every region ≤ 5 % of its unsuppressed signal, and stir only where fat_sat leaves > 3 %
(`fat_suppress/task.py::score`). The target scan costs 196–390 s; the budget is 120 s. Reference
(`fat_suppress/controls.py`): fat_sat pair in the target geometry at TR 60 ms, then four signed IR probes at short TR,
fit, null carried to the target TR.

**1. Classes, 12 episodes read** — (b) 8, (c) 4.
- (b) probes at the target TR, budget gone after a few scans, then an unmeasured answer: L1/s9000/e5, L1/s9002/e0,
  L1/s9003/e2, L2/s9000/e3, L2/s9001/e7, L2/s9002/e0, L3/s9002/e5, L3/s9003/e1.
- (c) right plan, sloppy: L1/s9001/e2 (pair and IR at TR 500, then ignores its own numbers), L2/s9003/e2 (7 IR probes,
  nulls off by 25–120 ms), L3/s9000/e5 (5 of 6 IR probes refused for budget), L3/s9001/e6 (methods all right, one IR
  probe, then TI 270 "typical" for all).
- Snippets: L1/s9000/e5 "Full target scan won't fit (319 s). I'll use cheap partial-Fourier scans" — still at TR 2450,
  76.5 s each — then "the defensible answer is to deliver fat_sat for all regions".
  L2/s9001/e7: "32 lines × 1800 = 57.6 s + 12 = ~70 s, too much … I have only 28.8 s".
- Whole-run counts (`sfc/feat_fat.py`), 94 failing submissions: 70 spend ≥ half their scan time at ≥ 0.8 × target TR, 45
  scan only there; 62 never run an IR scan; 32 never scan with fat_sat; 33 deliver fat_sat for all regions; 55 deliver
  fat_sat where it leaves > 50 %; 8 deliver TI 264 ms, the datasheet's STIR preset.

**2. Statement vs verifier.** Both rules are stated and match `score`; no hidden requirement found.
- Rule 2 is read backwards at least once: L1/s9001/e2 "R3's fat-sat residual is only ~29% — nowhere near the 3%
  threshold that would force IR". 8 episodes deliver fat_sat against their own pair measurement (> 20 % left).
- Near misses by hand: L2/s9002/e6 (the pass) TI 210/260/260/225/210 against oracle 211/279/275/228/212, residual 0–5 %.
  L2/s9001/e3: R3 TI 250 against 291 → 10 %, R6 TI 320 against 230 → 23 %, three regions pass.
  L2/s9000/e0: one TI 280 for four regions, right for R2 (279, 0 %), 9–24 % on the others.
**3. Distance.** Nowhere near. Median 4–5 of 5 regions fail; no failing episode has one failing region or a worst residual
≤ 10 %. Of 159 failing stir regions none is within 15 ms of the oracle null, 26 within 15–40 ms; median error 67–80 ms.

**4. One change.** None converts many: the gap is competence (measurement design, then the null). Cheapest help: the
sentence ArtefactClinic already has ("the budget is smaller than one run of the target scan: experiments have to be
cheaper"). Expect little: with that sentence Clinic still has 14 (L1) and 20 (L2) of 32 episodes entirely at the card TR.

**5. Hint tier.** (1) fat_sat does not depend on TR: run the target matrix and bandwidth at TR 60 ms with and without
it, full phase FOV; (2) four IR scans at TR ≈ 600 ms, matrix 48, TI 60–500 ms, read the sign from the phase, fit
a − b·e^(−TI/T1) per region; (3) carry the null to the target TR. Not tested here.

**6. Defect (interface).** A reduced `phase_fov` folds unsuppressed material onto a region and the read-out gives no
sign of it. Verified on the engine for L1 s9001 R3: fat_sat/plain = 0.021 in the target geometry, 0.022 at matrix 64,
0.013 with partial Fourier 0.6, 0.297 with `phase_fov` 0.5 (what L1/s9001/e2 measured: 0.29). 67 of 96 episodes use
reduced-phase-FOV probes; 5 matched pair readings are 22–39 % where the verifier sees 1–8 % (only this one traced).

## 4. ArtefactClinic (0/96)
Pass needs every finding's cause set exact, every finding clean on the delivered protocol, and all requirements
(`artefact_clinic/task.py::score`). Truth is known to me only for the 7 cells that rebuild (56 episodes, section 0).

**1. Classes, 12 episodes read** — (b) 9, (c) 2, (d) 1.
- (b) too few or wrong differential experiments: L1/s9003/e7 (TE only, 2 scans), L1/s9001/e7 (fat_sat on/off and a TR
  change, all "object"), L1/s9000/e4 (TE only → all opposed-phase; remedy fat_sat, a locked field), L2/s9001/e1 (ringing
  finding called wrap; 193.5 s against 115.9 s), L2/s9002/e0 (all four wrong), L3/s9000/e7 (ghosts found, wrap never
  tested), L3/s9001/e6 and L3/s9003/e4 (no half-slice scan; e6 also delivers TE 3.6 ms at 3 T, still opposed),
  L3/s9002/e2 (one scan → all "object").
- (c) L2/s9000/e2 (diagnosis 4/4, clean 4/4; `averages` left out → SNR 2311 against 4357.3), L2/s9003/e7 (diagnosis 4/4;
  TE 3.5 instead of 4.6 ms in-phase at 1.5 T → F1 5.6 % against 5 %).
- (d) L1/s9002/e3: names causes from the look of the image and delivers the complaint protocol unchanged.
- Snippets: L1/s9000/e4 "Budget can't change readout resolution (matrix must stay ~112). So I can't directly test
  truncation ringing." L1/s9003/e7 "F2 barely changed — likely the object itself. I have 20.5 s left, not enough for
  another full-TR scan."
- Whole-run counts (`sfc/feat_clinic.py`): a doubled-matrix scan in 4 of 96 episodes, a thinner-slice scan in 0; all
  scans at the card TR in 14 / 20 / 32 of 32 (L1/L2/L3); 0–1 kinds of experiment in 19 / 15 / 28 of 32.
  On the verified cells truncation is named for 3 of 24 truncation findings, susceptibility for 1 of 40; opposed-phase
  findings get "chemical shift + opposed phase" in 9 of 24 at L1. 14 episodes deliver the complaint protocol unchanged.

**2. Statement vs verifier.** Checks are stated. Two soft spots:
- `check` plans the delivered protocol as written, so a field left out takes the console default, not the card value;
  the text does not say so. 2 episodes omit `averages` (L2/s9000/e2, where it is the only failing check; L2/s9002/e3).
- L3 SNR floor is "86 % of … the protocol with the highest SNR that meets everything else": a number the card does not
  print (`task.py::card` says so itself).
- 19 `locked_parameters` failures are plain violations of a stated list: `tr_ms` 8, `fat_sat` 7, `ro_polarity` 4.
- Near misses by hand: L2/s9003/e7 5.6 % against 5 % (1.12 × tolerance); L2/s9000/e2 SNR 0.53 × the floor, and keeping
  8 averages would have broken the 108 s limit (as L2/s9000/e6 did, 166.8 s); L1/s9003/e7 one label wrong, all else met.
**3. Distance.** L3 nowhere near. L1: best is 3 of 4 labels (4 episodes; L1/s9003/e7 meets everything else). L2: three
episodes with 4/4 diagnoses lost on requirements (L2/s9000/e2 SNR; e6 locked + scan time; L2/s9003/e7 clean by 0.6 points).

**4. One change.** For conversions none: the truncation and susceptibility experiments are almost never run (doubled
matrix in 4 of 96 episodes here, 5 in c01; thinner slice in 0 here, 9 in c01). c01 passes 9/32 at L1 and 4/32 at L2;
all 13 passing answers use only object / opposed-phase / wrap, and no c01 answer names truncation. For validity: fix
the concurrent instance build (section 0) before any training use.

**5. Practice level / hint tier.** Practice: findings from {opposed phase, object} only (L2: + wrap). Hint tier: one
experiment per cause — in-phase echo (4.6 ms at 1.5 T, 2.3 ms at 3 T) removes opposed-phase cancellation, doubled matrix
changes ringing, more phase FOV removes wrap, more bandwidth shrinks displacement, half slice restores susceptibility
loss; unchanged by all = the object; name only what an experiment changed; restate every protocol field. Not tested.

**6. Defect.** Section 0 (instance generation under threads).

## 5. Not read, not verified
- 37 conversations read (13 + 12 + 12), most as narratives (assistant text, scan calls, read-outs), three in full.
- c01 conversations not opened, only records and scan lists; c01 has no FatSuppress run. Clinic truth for the 5
  mismatching cells (40 episodes) is unknown. No reference control re-run; the hints in items 5 are untested.
