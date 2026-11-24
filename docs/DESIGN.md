# Design notes

## Definition

A physics-application episode: the agent faces a hidden *system + tissue* world through a protocol-level scanner, has a budget
too small for search, and must deliver a quantity or a protocol that is wrong unless a specific physical confounder is
recognised and handled. Reward is the error of what was delivered, measured on the world.

## Architecture

1. **Scanner service.** One process per episode, a 128-bit secret seed that appears in no agent-visible file, owner of the world,
   the noise stream and the scan-time ledger. No reset route, no truth route. The log is append-only and hash-chained; the
   verifier replays it.
2. **Two calls.** `plan` is free and returns the realised protocol, scan seconds and limits as percent of limit. `scan` is
   charged and returns console-level images and read-outs on regions the agent chooses.
3. **World.** Tissue (PD, T1, T2, T2′) on analytic shapes, smooth fields (B1+, ΔB0, receive profile), system state
   (transmit-gain error, centre-frequency offset). Mechanisms are documented for the agent; magnitudes are hidden.
4. **Engine tiers on one world definition.** Tier 1: closed forms. Tier 2: EPG steady state per tissue class and B1 bin.
   The served and scored signal uses tier 2 where the two differ. A consistency suite asserts both where the tiers must agree
   and where they must differ.
5. **Admission by control policies.** Per instance: the reference policy passes with margin; no-scan prior, confounder-blind
   textbook protocol, best fixed recipe, brute force within budget, wrong units and seed attack fail. A family that a fixed
   recipe passes is rejected.
6. **Reward.** Hard gates (valid deliverable, scans present in the service's log, units), then a continuous score of the final
   physical error that is exactly 0 at the no-scan prior policy and 1 at or below a tolerance measured per instance from the
   reference policy's error distribution. Process quantities are logged, not paid.

## Why `t1_transmit` is rejected

One constant protocol — a B1 map, two flip angles and a joint fit, tuned on other worlds — passes nearly every instance, so the
admission rule admits none (`python3 -m mrfrl.t1_transmit selfcheck` prints the table). The family forces a correct procedure
but no decision that depends on the instance.

## Splits and contamination

Training tasks are generated here and share no task, generator or reference solution with MRI Agent Bench, which stays a
held-out evaluation. In `recon_repair`, three mechanism groups (line order, phase axis, receiver gain) are never injected in
the train split.
