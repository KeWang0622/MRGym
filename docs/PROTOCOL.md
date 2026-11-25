# What a reportable run is

A number about an agent on MR Gym is reportable when another person can recompute it from saved records.

1. **Name the three versions**: package (`mrgym --version`), scanner engine, environment id with its `-vN`.
2. **Fix the instances before the run**: environment, levels, and the seed list of the public train split
   (`mrgym run-agent --seeds 0-23`). Run every agent on the same instances; `mrgym results --check` fails when the
   seed sets of the rows of one table differ.
3. **Fix the harness**: tool set (with or without `calc`), the tool-call limit, sampling parameters, reply limit,
   system note. Report them. A changed setting is a new run folder, never a continuation.
   - **One set of limits, and the agent is told each of them.** An episode ends at the terminal tool or at a limit of
     the environment: scan time, tool calls (`max_steps`), rejected calls. The task text, the status line under every
     result (the tool calls under every one; the scan time left and the rejected calls whenever they changed), the `status`
     tool and `info["limits"]` state the same numbers. A harness adds no cap of its own: the
     reference runner's `--max-turns N` makes the environment with `max_steps=N` and prints it, and its only other
     stop is three replies in a row without a tool call. A harness limit the agent can hit (the reply length) is
     passed to `env.reset(options={"harness": {"max_reply_tokens": N}})` so that the task text names it.
   - **Report the interface tax next to every model row**: `mrgym iface-tax --dir RUN` gives the share of calls and of
     turns that were refused for how they were written (error classes `shape` and `vocabulary`), with the refusals of
     the instrument, the calls the canonicaliser rewrote and the replies cut at the reply limit beside it. A model
     that loses more than one call in ten this way is interface-limited: its pass rate measures the format, not the task.
   - **Scripted policies** read images in process (`ToolScanner.view` hands them the array and the extent of its
     bright pixels from the trainer's side). No tool returns that extent; a model measures with `profile`. Say which
     of the two an agent row used.
4. **Count every started episode.** An episode that ends without a submission (a limit of the environment, no tool
   call) scores 0 and stays in the denominator. Only infrastructure failures before a verdict (endpoint down, HTTP errors
   after retries) are excluded; they are counted in their own column and re-run.
5. **Report rates as `k/N (p%)`** with the interval method, and mean reward and median error next to the pass
   rate. Report effort: scans, share of the scan budget, turns, tokens.
6. **Keep the scripted rows next to the model rows**: reference, no-scan prior, best constant recipe and the other
   control policies on the same instances. A model row without them cannot be read.
7. **Say what the instances are**: the environment id, the levels and the split. Every instance of a level is
   served; the gate gives one verdict per level (`mrgym list`), so say whether a level is inside the admitted claim,
   a candidate or a practice level.
8. **Keep the records**: one `record.json` per episode, the messages for model episodes, the scanner log.
   `mrgym results --check` must pass on the folder that is published.

Maintainer-run results are labelled as such. Results on public training seeds are not held-out results.
