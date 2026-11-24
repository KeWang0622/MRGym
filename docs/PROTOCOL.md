# What a reportable run is

A number about an agent on MR Gym is reportable when another person can recompute it from saved records.

1. **Name the three versions**: package (`mrgym --version`), scanner engine, environment id with its `-vN`.
2. **Fix the instances before the run**: environment, levels, and the seed list of the public train split
   (`mrgym run-agent --seeds 0-23`). Every agent plays the same instances.
3. **Fix the harness**: tool set (with or without `calc`), turn cap, sampling parameters, system note. Report them.
   A changed setting is a new run folder, never a continuation.
4. **Count every started episode.** An episode that ends without a submission (turn cap, no tool call, truncation)
   scores 0 and stays in the denominator. Only infrastructure failures before a verdict (endpoint down, HTTP errors
   after retries) are excluded; they are counted in their own column and re-run.
5. **Report rates as `k/N (p%)`** with the interval method, and mean reward and median error next to the pass
   rate. Report effort: scans, share of the scan budget, turns, tokens.
6. **Keep the scripted rows next to the model rows**: reference, no-scan prior, best constant recipe and the other
   control policies on the same instances. A model row without them cannot be read.
7. **Say what the instances are**: whether they include instances the admission gate rejects
   (`info["admitted"]` at reset, `mrgym.admission.admitted_seeds`).
8. **Keep the records**: one `record.json` per episode, the messages for model episodes, the scanner log.
   `mrgym results --check` must pass on the folder that is published.

Maintainer-run results are labelled as such. Results on public training seeds are not held-out results.
