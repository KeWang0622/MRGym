# Security

MR Gym pays an agent for a result that is correct on a hidden world. A way to be paid without measuring is a
security problem of this project, and it is handled like one.

## Report privately

Use [a private advisory](https://github.com/KeWang0622/MRGym/security/advisories/new). Do not open a public issue
for anything in the list below.

Include the output of `mrgym --version` (package and engine), the environment id and level, and the smallest
script that shows the problem.

## Useful reports

- A way to obtain reward without the measurement.
- A way to read the hidden world or the seed from anything an agent can reach: a tool reply, a status line, a task
  id, an error message, a file, the service.
- A way to reset or exceed the scan budget or the request limits.
- A way to alter the scan log, or to make an altered log replay.
- An ordinary vulnerability of the HTTP service or its client, such as a request that makes the service read or
  write outside its log directory.

A fixed exploit becomes a control policy of the environment it affected and is named in the changelog.

A policy that reaches a high reward by a legitimate physical shortcut, such as one constant protocol that solves
every instance, is not a security problem. Open a normal issue; it becomes a control policy.

## Trust model

| Surface | What it hides the world from | What it does not hide it from |
|---|---|---|
| In process (`mrgym.make`, `env.step`) | a model that only sends tool calls | any code that runs in the same process |
| HTTP service (`mrgym serve`), one process per episode | an agent in another process that can only reach the port | anyone who can read the service's files, memory or episode config |

In-process environments hide the world from a tool-calling model, not from code running in the same process.
Python has no private memory: code in the process can read the seed, the world and the score function.

An agent that can execute code needs the HTTP service behind a process boundary: the service in one process or
container with the episode config and the log, the agent in another with the port only. The service has no route
that resets, re-seeds or returns anything derived from the seed other than measurements. That boundary has been
exercised by tests on one machine; it has not been reviewed by a third party.

Training seeds are public. Without `MRGYM_SECRET` every world can be rebuilt from its seed with the public
generator, which is what a training split is. A held-out split refuses to run without a secret.

`extras/recon_repair/` is outside this model. Its README states its own limits.
