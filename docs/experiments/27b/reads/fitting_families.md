# Why thinking-off Qwen3.8-27B fails four fitting families, beyond the calc limits (run c04)

Date 2026-11-25. Read-only analysis. Episode paths are relative to
`runs/c04/records/cal/episodes/qwen3.8-27b/Scanner-<Family>-v<N>/`.
Scripts, digests and hidden-truth tables: `analyst-scratch/`
(`table.py`, `nt_feat.py`, `t1_feat.py`, `bf_feat.py`, `ps_feat.py`, `truth.json`, `dig_*.txt`).

## 0. What was and was not read
- Read (digest of every turn: all scans + read-outs, every refusal, narration, last calc code, final reasoning):
  NullTissue L0 8 passes + all 9 submitted failures, L1 all 23 submitted failures (one in full), L2/L3 9 failures + 1 pass;
  T1Transmit 14; BandFree 10; ProtocolToSpec 10. Statistics below are over all `record.json` / `scanner_log.jsonl`.
- Hidden truth: the oracle (`null_tissue/oracle.py: solve`) run on all 16 NullTissue instances (null and pass window in ms),
  `t1_transmit/task.py: true_t1_ms` on all 12; five NullTissue near misses re-scored with `task.score` (all agree with the records).
- Not read: the no-submission episodes; narration beyond the first 125-260 characters of non-final turns; calc code of non-final
  calls; `controls.py` bodies of T1Transmit / BandFree / ProtocolToSpec (docstrings only) and their oracles; c01 conversations
  (c01 scan logs and outcomes only). Classes for unread episodes are not claimed.

## 1. Table from all records (32 episodes per cell = 4 instances x 8)
"First failing check" is the first False in `checks` order (R1 before R2 ...), not the first in time.

| cell | pass | submitted | not subm. | first failing check (failed-submitted) | distance to passing |
|---|---|---|---|---|---|
| NullTissue L0 | 10 | 19 | 13 | nulled_R1 9 (R2 also 7) | worst_ratio min 0.080 / med 0.46 (limit 0.05); worst miss 1.6 / 7.8 half-windows; within 2x: 1 of 9 |
| NullTissue L1 | 0 | 23 | 9 | nulled_R1 23 (R2 23) | 0.135 / 0.59; miss 2.8 / 13.7; within 2x: 0 of 23; regions inside window 1 of 46 |
| NullTissue L2 | 1 | 22 | 10 | nulled_R1 21 | 0.086 / 0.63; miss 1.7 / 11.8; within 2x: 1 of 21; regions inside 1 of 63 |
| NullTissue L3 | 0 | 24 | 8 | nulled_R1 21, R2 3 | 0.055 / 0.61; miss 1.1 / 13.1; within 2x: 2 of 24; regions inside 14 of 72 |
| T1Transmit L1 | 1 | 16 | 16 | R1 9, R2 3, R4 2, R5 1 | worst ln-ratio / ln 1.1: min 1.19, med 8.0; within 2x: 1 of 15; regions within 18 of 79 |
| T1Transmit L2 | 0 | 18 | 14 | R1 16, R2 2 | min 1.02, med 14.0; within 2x: 1 of 18; regions within 12 of 100 |
| T1Transmit L3 | 0 | 22 | 10 | R1 20, R2 1, R4 1 | min 1.05, med 11.4; within 2x: 1 of 22; regions within 11 of 128 |
| BandFree L1 | 0 | 25 | 7 | band_free_A 19, bandwidth_low_A 5, acquisitions_needed_A 1 | 1 - worst kept fraction (pass <= 0.2): 0.09 / 0.62 / 0.90 |
| BandFree L2 | 0 | 24 | 8 | band_free_A 21, bandwidth_low_A 2, acquisitions_needed_A 1 | 0.30 / 0.74 / 0.93 |
| BandFree L3 | 0 | 24 | 8 | band_free_A 21, acquisitions_needed_A 2, bandwidth_low_A 1 | 0.08 / 0.77 / 0.96 |
| ProtocolToSpec L1 | 0 | 23 | 9 | t1_contrast_A 19, snr_A 4 | error = -min ln(achieved/required), pass < 0: 0.54 / 1.94 / 4.68 |
| ProtocolToSpec L2 | 0 | 28 | 4 | t1_contrast_A 15, t2_contrast_A 12, snr_A 1 | 0.31 / 1.05 / 3.99 |
| ProtocolToSpec L3 | 0 | 28 | 4 | t1_contrast_A 21, snr_A 4, t2_contrast_A 3 | 0.22 / 1.45 / 2.87 |

NullTissue pass window (oracle): +-7 to 8 % of the null (e.g. L1 s9000 R1 null 239.9 ms, window 222.8-257.9).

## 2. NullTissue: L0 -> L1
**Statement diff** (`null_tissue/task.py: hint_text`). L1's text is L0's minus two paragraphs; budget 150 s vs 185 s of which 35 s is
already spent on the given probe, so the scan time left is the same 150 s.
1. "How to do it": (1) probe with `ir` at a short TR "for example tr_ms 800, matrix 64, partial_fourier 0.6", three or more TI from
   60 ms to TR - 50; (2) sign rule: at the longest TI every signal is positive, a read-out whose phase differs from it by more than
   pi/2 is negative; (3) fit S(TI) = a - b exp(-TI/T1), q = b/a; (4) carry-over: k = (q-1)/(1 - q E(TR_probe)),
   TI_null = T1 ln((1+k)/(1+k E(TR_target))).
2. "The first probe is already acquired" (TR 800, TI 740) with its read-out and the sentence that every signal in it is positive.

**What the model does with it (L0).** All 19 submitted L0 episodes probe at TR 800 (one adds 600, one 200), 3-7 IR scans, full phase
field of view, sign from the given probe. The 8 passes read (L0/s9000/e4, s9001/e2, e4, e5, s9002/e3, e5, s9003/e4, e6) execute
steps 1-4 literally and land within 0.0-0.55 half-windows.

**What kills L1** (all 23 submitted failures read):
- **Step 1 is missing: 16 of 23 spend the budget at TR 6000.** 14 probe with `ir` at the target TR (or 12 000): L1/s9000/e1, e2, e5,
  e6; s9001/e0, e2, e3, e4, e7; s9002/e0, e1, e2; s9003/e3, e6. One probe costs 57-129 s of 150 s, so they hold 1-2 TI points
  (12 of 23 have <= 2 IR scans; median 2 vs 5 at L0) and then guess. Two more burn 114-117 s on `se` at TR 6000 (s9003/e0, e7).
  To afford TR 6000 they cut the phase field of view to 0.25, which leaves regions outside the image: 8 of 23 paid for at least
  one scan whose read-out was `R1 0 - - - -` (e.g. s9001/e3 117 s, s9001/e4 105 s, s9003/e5 five scans).
- **Steps 3-4 are missing: the 7 that do probe at short TR all deliver without a valid fit or carry-over**: zero crossing or two-point
  magnitude fit at the probe TR delivered as it is (s9000/e3, s9001/e1, s9003/e2), never sampled below the null and concluded
  "no null in range" (s9002/e6 -> 5984 ms), TR 200 steady state misread (s9003/e4 -> 6.04 ms), T1 from a 2-angle 2D SPGR
  (s9000/e0: 117 ms, true 351), region outside the image (s9003/e5).
- Step 2 (sign) is decisive less often but visible: s9000/e5 reads phases -0.81 / -0.93 as "both vials negative" (both were positive).
- With full reasoning (c01, older interface) L1 passes 3 of 18 submitted; 6 of its 15 failures still probe at TR >= 5000.

Quotes. L1/s9000/e1 t3: "I'll measure T1 with a low-cost reduced-resolution IR sweep at the mandated TR (6000) so the TR effect is
identical"; t13: "The IR is locked to TR 6000 by the task". L1/s9001/e1 t11: "the null condition 1-(1+k)E(TI)=0 is independent of
TR and T2 (they cancel), so the TI I measured with short-TR probes is valid for the TR=6000 target scan" (delivered 336 ms, null 418).

## 3. Classes of the first decisive error (episodes read)
(a) interface wall, (b) wrong method, (c) right method executed badly, (d) never engages, (e) budget, (f) fallback after refused calc.

**NullTissue L0, 9 of 9 failures.** The recipe is followed up to the fit; the fit breaks and the model falls back.
- (c) 4: s9001/e1 (closed form with d**360, T1 negative, delivers 5611 ms), s9002/e7 (own least-squares sign bug), s9003/e5 (reads
  the TI 380 probe as negative although its phase equals the reference), s9003/e7 (probes at TR 600 against the TR 800 reference).
- (f)-driven 5: s9000/e1, s9000/e6, s9001/e0, s9001/e7, s9003/e1. The step limit removes the T1 grid search, the model switches to
  two/three-point closed forms or a grid too short to contain T1, and stops trusting the fit. A stricter count (the last calc
  before `submit` was refused) gives 0; "refusal in the last 4 calc calls and late text blames calc" gives 1.
- End state shared by 5 of 9 (s9000/e1, e6, s9002/e7, s9003/e5, e7): the zero crossing at the probe TR is delivered untransferred.
**NullTissue L1, 23 of 23.** (b) probe at target TR 14; (e) budget on non-IR scans 2; (b) short-TR probes, no fit/carry-over 7
  (paths in section 2). Co-occurring (a): 8 with a zero-pixel read-out. (f): 0 strict; calc limits visible in s9003/e2, s9001/e7.
**NullTissue L2/L3, 9 read.** Probing at the target TR is right here (19 of 21 and 19 of 24 do). (c) 6: two or three points, then
  linear interpolation over 200-600 ms or a 2-point fit with an assumed inversion efficiency: L2/s9001/e4, L2/s9001/e3,
  L2/s9003/e1, L3/s9000/e4, L3/s9001/e3, L3/s9003/e4. (b) 2: L2/s9000/e5 (T1 from SPGR), L2/s9001/e6 (TR 8000 probe, mixed TRs).
  (a)+(e) 1: L3/s9001/e0 (two 93 s scans with fov 80 mm, all regions outside, then guess). Zero-pixel read-outs: 10 of 21, 9 of 24.
**T1Transmit, 14 read.** (b) 12, (c) 2, (f) 0.
- (b) nominal flip angle, no B1 measurement, 7: L1/s9000/e3, L1/s9000/e4, L1/s9001/e4, L2/s9001/e2, L2/s9002/e6, L2/s9003/e2
  (delivers 30 ms for all six regions), L3/s9001/e6.
- (b) spin echo at 2-3 TRs with S = M0 (1 - E), assumed flip-independent, 5: L1/s9000/e6, L1/s9002/e0, L1/s9003/e0, L3/s9003/e1,
  L3/s9002/e6. Near the truth on the three instances read with transmit gain 0.88-1.14, off by 24-39 % at 0.76 (L3/s9002/e6).
- (c) 2: L1/s9003/e4 (plain SE ratio gave 459/479/543/350 ms against 456/478/580/332, then a mis-read AFI "88 degrees" correction
  moved all five out), L2/s9000/e5 (AFI abandoned, SE at TR 90/300).
Quotes. L2/s9002/e6 t4: "`excitation: nonselective` makes every voxel see the *nominal* flip angle (uniform, no unknown
position-dependent factor)". L1/s9000/e6 t11: "The SE method is robust to the flip-angle error because the 90/180 steady-state signal
equals M0(1-E)e^(-TE/T2) regardless of the partial 90 angle."
**BandFree, 10 read.** (b)/(d) 10, (f) 0. No episode maps the field (1 of 73 runs a multi-echo spgr) and none evaluates the 20 Hz drift.
- allowed-minimum (or near) bandwidth taken for rule 3 without testing rule 1: L1/s9000/e0, L1/s9000/e7, L1/s9001/e4, L2/s9001/e3,
  L3/s9001/e5. Over all 73 submitted: 29 deliver the minimum bandwidth for an image, 40 deliver <= 120, 41 the same protocol twice.
- a band-free phase cycle found at one bandwidth (240-600) and no search downwards: L1/s9002/e2, L1/s9002/e6, L1/s9003/e2, L2/s9003/e0.
- an unneeded pair: L3/s9000/e6 (35 of 73 deliver two phase cycles for an image).
Quotes. L2/s9001/e3 t13: "Phase cycle 180 with the lowest bandwidth (60 Hz/px) and console-minimum TR (19.96 ms) clears every
marked region with margin at the current center frequency. Submitting for both images." L1/s9003/e2 t11: "A single acquisition at
the lowest allowed bandwidth (600 Hz px) with phase cycle 0".
**ProtocolToSpec, 10 read.** (e)/(d) 6, (a)+(e) 1, (c) 3, (f) 0.
- one scan at matrix 64-128 costs 56-59 s of 63 s, no second scan, TI/TE from typical values: L1/s9000/e6, L1/s9001/e7, L1/s9003/e3,
  L2/s9003/e7, L2/s9002/e1, L3/s9001/e5. Over all 79 check_failed: 55 made exactly one scan; 45 deliver one protocol for both images.
- strip field of view missed 4 of 6 regions twice: L3/s9003/e3.
- two scans as designed, then the solve is wrong: L1/s9002/e6 and L1/s9000/e2 (SNR not scaled from matrix 32 to 128), L2/s9000/e5
  (T2 measured, TE not solved for the 1.5 ratio).
Quotes. L1/s9000/e6 t4: "I have 63 s budget. Let me measure the standard protocol with fewer lines (matrix 64) - cost ... 56 s";
t7: "I'm out of scan time (7 s left). Let me reason analytically". L1/s9002/e6 calc 22: "Estimated SNR_R2 at ti=70: 51.5 ... All
requirements met at ti=70, te=12, averages=1" (verifier: SNR 11.6).

## 4. How far from passing; near misses checked by hand
- **NullTissue**: nowhere near except L3. Episodes with every region within 2 half-windows: 1 / 0 / 1 / 2 (L0-L3).
  L1/s9001/e7: R2 482.0 ms against a window of 482.09-558.93 (ratio 0.0501, limit 0.05), R1 508 against 388-450.
  L2/s9001/e4: 724 / 516 / 735 against nulls 648 / 480 / 673 (keeps 8.6 / 5.2 / 7.0 %): two probes, T1 right to 3 % for R2, then
  0.6931 x T1 at TR 2330 with efficiency 0.957. L3/s9001/e3: 409 / 675 / 774.5 against windows 345-406 / 588-689 / 733-857:
  linear interpolation between TI 300 and 900 misses R1 by 3 ms. All three re-scored: the verifier is right.
- **T1Transmit**: one episode per level within 2x tolerance, the rest 8-14x. L2/s9000/e5: 0.96 / 0.908 / 1.05 / 1.07, R2 at -9.2 %
  against the -9.09 % edge. L3/s9003/e1: 1.07 / 0.96 / 1.00 / 1.105 / 1.06, R4 at +10.5 %. L1/s9002/e0: 5 of 6 within, R4 -10.7 %.
  All three are spin-echo multi-TR episodes on instances with a small transmit error.
- **BandFree**: nowhere near. 6 of 73 are band-free on both images; all 6 then fail rule 2 or 3 by a wide margin: L1/s9002/e2
  240 Hz/px against a lowest of 141.79 (1.69x, limit 1.15x), L1/s9002/e6 300 (2.12x), L1/s9003/e2 600 against 328 (1.83x).
- **ProtocolToSpec**: nowhere near; no episode has every requirement within 20 %. L1/s9002/e6: both contrasts pass, SNR 11.6 of
  20 with 1 average; re-scored with 3 averages it passes (SNR 20.16, 368 s of 482 s), with 4 as well: one integer from a pass. L3/s9003/e3: SNR 19.2 and 13.2 of 20.
  L2/s9003/e7: T2 contrast 1.44 and 1.06 of 1.5.
- **Verifier**: no defect found in `score` of any of the four. T1Transmit's printed "-9.2 %" failing while "+9.9 %" passes is the
  stated factor rule (T1/1.1 to 1.1 T1).
- **Text and presentation ambiguities**:
  1. NullTissue statement: "The scan is the same for all of them except for its inversion time: sequence ir, tr_ms 6000 ..." and
     "Use the scan-time budget for whatever you need to measure first" never say that probes may use another protocol; 16 of 23
     L1 episodes treat TR 6000 as binding on their own scans (quotes in section 2).
  2. Datasheet topic `slice_profile`: "With `excitation: nonselective` every voxel sees the flip angle a" - `a` is read as the
     nominal angle. 11 of the 55 failed-submitted T1Transmit episodes use non-selective spgr; two read (L2/s9002/e6, L1/s9000/e3).
  3. Region read-out `phase_rad` is absolute with a per-world offset (the positive sign sits at -1.0 rad in one world, +2.4 in
     another); `scanner_info ir` says only "the sign is in the phase image". L1/s9000/e5 and L1/s9001/e2 read the number's sign.
  4. A scan can be paid for although no marked region is in the image; `plan` does not say so. Zero-pixel read-outs in 8 / 10 / 9
     failed-submitted NullTissue L1 / L2 / L3 episodes.
  5. ProtocolToSpec: after a first scan `plan` prices a deliverable protocol with the 3 s repeat overhead while the verifier charges
     30 s each (the statement does say "as a scan of its own"). Two `time` failures equal exactly that difference (L1/s9001/e0
     316.8 s of 295; L2/s9003/e3 470.4 s of 429); those two conversations were not read.
  6. NullTissue L0/L1 seeds draw inversion efficiency 0.926-0.979, so ln 2 x T1 overshoots the null by 2-6.6 % of a +-7.5 % window.

## 5. One change per family (no tolerance change, no method)
- **NullTissue**: one sentence in the statement: probe scans may use any sequence, repetition time, matrix and field of view; only
  the delivered inversion times are evaluated on the target scan. It removes the first decisive error of 16 of 23 L1 failures.
  It is not sufficient: the 7 L1 episodes that already probed at short TR passed 0 of 7. Second, pure interface: `plan` lists the
  marked regions outside the planned image (27 of 68 L1-L3 failures paid for such a scan).
- **T1Transmit**: none that converts many. The wording fix of ambiguity 2 is worth making; the dominant failure is skipping the
  B1 measurement, which is the method.
- **BandFree**: none. The model does not attempt the drift margin or the lowest-bandwidth search.
- **ProtocolToSpec**: `plan` prints what the budget holds after the planned scan ("leaves 7.0 s"). 55 of 79 failures take a
  first scan that excludes a second. Not sufficient: the 24 episodes with two or three scans also passed 0.

## 6. Cells expected in the 20-60 % band once class (f) is removed
- **NullTissue L0 only, and it is already there (10 of 32).** Submitted episodes pass 10 of 19; if the 13 rejected-call episodes
  submit at that rate and the 5 (f)-driven failures partly convert, 17-20 of 32 (53-62 %), the top edge of the band (c01 with
  reasoning: 21 of 32).
- **No other cell.** Passes among submitted: NullTissue L1 0 of 23, L2 1 of 22, L3 0 of 24; T1Transmit 1 of 16, 0 of 18, 0 of 22;
  BandFree 0 of 73; ProtocolToSpec 0 of 79. Strict (f) among submitted failures (last calc before `submit` refused) is 0-2 per
  cell, and the episodes that never submitted have no reason to do better than those that did.
