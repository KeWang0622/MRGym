# Analysis plan: how MR Gym numbers are computed, compared and stopped

This page fixes the analysis before any run that is meant to be published. The machine-readable plan of a run is
`analysis_plan.json` in its results folder (`mrgym results --register-plan`); it is written before the first episode,
every record stores its hash, and `mrgym results --check` fails when a record is older than the plan or when the
computed comparisons are not the planned ones. The code is `src/mrgym/stats/` (numpy and scipy; no random numbers on
the primary path, so every stored number can be rebuilt byte for byte).

## 1. The unit is the instance

An instance is an environment, a level and a seed. Episodes of one instance are repeats, not new units.

A pass rate is printed with the interval of `mrgym.stats.cluster_rate`: the Wilson interval on the Korn-Graubard
effective sample size with a t quantile.

```text
instance mean    ybar_i = mean of the K_i episodes of instance i
estimate         p      = sum_h w_h * mean_{i in h} ybar_i               strata h = levels, weights w_h = 1 / H
variance         v      = sum_h w_h^2 * s_h^2 / M_h                      s_h^2 = sample variance of ybar_i in stratum h
effective size   n_eff  = min(episodes, p (1 - p) / v);  p in {0, 1} or v = 0:  n_eff = M (instances, not episodes)
interval         Wilson(k = p * n_eff, n = n_eff) with the t quantile on M - H degrees of freedom
design effect    deff   = episodes / n_eff
p = 0            one-sided upper bound 1 - 0.05^(1/M);     p = 1: one-sided lower bound 0.05^(1/M)
```

Means of continuous quantities (reward, tokens) get a t interval on instance means with the same strata
(`cluster_mean`). A cell with fewer than 100 instances also gets the Beta-binomial credible interval of the clustered
model (uniform prior on the rate, Gamma(1, 1) on the concentration, integrated on a fixed grid), and the wider of the
two is printed. Normal-approximation and bootstrap intervals are not used: they are too narrow on small samples and
have zero width when every episode fails.

## 2. The design of a published row

- **Cell.** One agent on one environment. Strata are the levels, with equal allocation.
- **Size.** 300 instances per environment and 2 episodes per instance for a model. A scripted policy is
  deterministic and plays one episode per instance.
- **Why two episodes and not more.** Instances are generated, so a new instance costs what a repeat costs. The
  variance of a cell mean is `p (1 - p) (rho + (1 - rho) / K) / M`: for a fixed number of episodes, one episode per
  instance is the most precise design. Two is the smallest number that also gives retest agreement, the
  between-instance share of variance and pass^2.
- **Instances.** The `report` suite ([results/manifests/report.json](../results/manifests/report.json)): string seeds
  `report:<i>`, disjoint by prefix from the seeds of the gate (`gate:<i>`), of tuning (`tune:<set>:<i>`) and of
  checkpoint selection (`val:<i>`). Every agent plays the whole list; a row with another list is another table.
- **Sampling.** The temperature is the provider's documented default, sent explicitly and recorded. It is not tuned,
  and it is not set to zero to tighten an interval.
- **Tiers.** A cell with fewer than 100 instances, or with records that do not store the conversation, is `pilot`: it
  is listed under its own heading and never enters a comparison.

What the designs detect, in points of pass rate (worst case p = 0.5, two-sided alpha = 0.05, power 0.80; `rho` is the
between-instance share of variance, `r` the correlation of the two agents' instance effects):

<!-- POWER:START -->
| Design (instances x episodes) | 95 % half-width of one rate | Detectable paired difference, rho = 0.07, r = 0 | same, Holm over 10 | rho = 0.3, r = 0.5 | Use |
|---|---:|---:|---:|---:|---|
| 10 x 1 | 35.8 | 62.6 | 81.6 | 57.8 | none |
| 24 x 1 | 21.1 | 40.4 | 52.7 | 37.3 | none |
| 96 x 8 | 4.4 | 8.7 | 11.4 | 9.9 | pilot for rho |
| 100 x 2 | 7.3 | 14.5 | 18.9 | 14.0 | descriptive |
| 300 x 1 | 5.7 | 11.4 | 14.9 | 10.5 |  |
| 300 x 2 | 4.2 | 8.4 | 10.9 | 8.1 | published minimum |
| 600 x 2 | 2.9 | 5.9 | 7.7 | 5.7 | trained against base |
| 1500 x 2 | 1.9 | 3.7 | 4.9 | 3.6 | suite score |
<!-- POWER:END -->

The default `rho` is the value measured on the open-model pilot run; `r` was measured once, between a model and a
model at the floor, and is taken as 0. The table has to be recomputed from a pilot of each new environment
(`mrgym stats power --rho RHO --r R`).

## 3. Endpoints

- **Primary:** the pass rate per environment.
- **Secondary:** mean reward; regret against the oracle (median and 95th percentile of the distance between the
  delivered answer and the optimum of the same instance, `mrgym results --oracle`); pass^2 (both episodes pass);
  per-level pass rates, which are descriptive: no comparison is computed from a per-level cell.
- **Cost:** total tokens in and out and total wall time per row; a dollar figure only from a dated price list with
  its source (`prices.json`).

## 4. Comparisons

Fixed before the run, in `analysis_plan.json`:

- every pair of models per environment, two-sided;
- every model against the strongest floor row of the environment (the blind policy with the largest upper bound in
  the stored gate result), one-sided: model above floor.

Each is a paired difference on the instances both agents played (`mrgym.stats.paired_compare`): the difference of
the instance means, its standard error and t interval with the levels as strata, the correlation of the instance
means, wins, losses and ties. The p-value is the exact McNemar test when both agents played one episode per
instance, and a t test on the differences otherwise. Multiplicity: Holm within an environment. A difference that is
not in the comparison table is not a finding, and overlapping or separate intervals of two rows are not a test.

## 5. Exclusions

An episode that ended in an endpoint or runner failure is not a result: it is replayed, counted in its own column,
and a row with more than 2 % of them is not published. Every other started episode counts, with reward 0 when
nothing was delivered. No instance is dropped: not for being rejected by a gate, not for an invalid deliverable, not
for low discrimination.

## 6. Reliability

`mrgym stats reliability` and the generated section of [results/README.md](../results/README.md) report, per agent and
environment: the between-instance share of variance (one-way ICC with its F interval), the effective sample size,
retest agreement and pass^2; per environment over the model panel: the separation of the agents, Cronbach's alpha
and the split-half correlation (only with five or more models that pass anything), dead instances, the classical
discrimination of each instance, the level effect from a logistic regression with cluster-robust errors, and the pass
share at half, one, one and a half and two times the tolerance. Item statistics are reported and never used to drop
an instance. No item-response model is fitted on a handful of agents.

## 7. The admission gate

One verdict per environment and level on fresh instances (`mrgym gate`); its statistics are
`mrgym.stats.family_gate`. Every condition must hold, each at a one-sided 95 % bound, so a family that truly violates
one condition is admitted with probability at most 5 %, without a correction over the blind policies. The reported
ceiling of the blind policies is corrected for their number. Operating characteristic (`mrgym stats power --gate`):

<!-- GATE_OC:START -->
```text
admission gate, alpha = 0.05: a must-fail policy is cleared when its one-sided upper bound is <= 0.1; the reference is accepted when its lower bound is >= 0.9 (Clopper-Pearson, exact binomial)

P(a must-fail policy is cleared | its true pass rate)
     n  cleared at      2 %      3 %      5 %      7 %     10 %     12 %
   100        <= 4    0.949    0.818    0.436    0.163    0.024    0.005
   300       <= 21    1.000    1.000    0.951    0.558    0.046    0.003
  1000       <= 84    1.000    1.000    1.000    0.961    0.049    0.000

P(the reference is accepted | its true pass rate)
     n       needs     90 %     95 %     97 %     99 %
   100       >= 96    0.024    0.436    0.818    0.997
   300      >= 279    0.046    0.951    1.000    1.000
  1000      >= 916    0.049    1.000    1.000    1.000

expected lower bound of the gap reference - ablation, reference at 98 %, independent outcomes (points)
     n   abl 50 %   abl 60 %   abl 70 %
   100       39.4       29.5       20.0
   300       43.1       33.1       23.4
  1000       45.3       35.3       25.5
```
<!-- GATE_OC:END -->

## 8. Stop rules

- **Checkpoint selection** uses the `validation` suite (200 instances, one episode each, sampling temperature) and
  never the reporting instances. The chosen checkpoint is then reported once on the `report` suite.
- **Looks across checkpoints.** At most J looks, declared in the plan. Each look is a one-sided paired test of the
  checkpoint against the base at alpha / J (`mrgym.stats.sequential.checkpoint_look`). Futility: stop when the
  one-sided upper bound of the paired difference at 1 - alpha / J is below the plan's smallest difference of interest.
- **Early stop inside one evaluation** (`mrgym.stats.betting_test`). Instances in manifest order;
  `x_i = (d_i + 1) / 2` with `d_i` the paired difference of the instance means; wealth
  `W_t = prod (1 + lambda_i (x_i - 1/2))` with `lambda_i` computed from the instances before `i` only. "A above B" is
  declared when the wealth reaches 1 / alpha. By Ville's inequality the error rate holds at any stopping time.

## 9. What is not done

- The correlation of instance effects between two capable models and the between-instance share of variance of the
  new environments are not measured; the power table uses the pilot values.
- The coverage of the clustered interval is tested by simulation at 100 instances and a between-instance share of
  0.3 only.
- No published row exists yet. The reporting instances are public seeds: anyone can rebuild the worlds, and a
  code-executing agent could too. Whether the published table moves to a secret split is the maintainer's decision.
- `mrgym run-agent` does not yet play the seeds of a manifest; until it does, a published row cannot be produced with
  the command line alone.
