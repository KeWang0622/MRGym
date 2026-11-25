# CI definitions

The workflows live here, not under `.github/workflows/`, because the token this repository is pushed with has no
`workflow` scope: GitHub rejects a push that creates or changes a file under `.github/workflows/` with such a token.
Nothing in this folder runs until the owner activates it.

| File | Jobs |
|---|---|
| `tests.yml` | `lint` (ruff, mypy baseline, zizmor); `test` on ubuntu, Python 3.10 to 3.13, from the built wheel; `test-os` on macOS and Windows; `lowest` (lowest declared numpy and scipy); `reference-stack` (golden digests bit for bit on numpy 2.2.6 / scipy 1.15.3); `quality` (`make audit`); `build` (wheel checks); nightly `fuzz` (the contract machine, random, 5,000 examples) |
| `release.yml` | on a published GitHub release: build and check without credentials, then publish to PyPI by trusted publishing with build provenance |
| `scorecard.yml` | OpenSSF Scorecard weekly and CodeQL |
| `requirements-audit.txt` | what `make audit` needs beyond the package, with the versions of the last audit |

Every action is pinned to a commit; the tag is in the comment next to it. The commits were read from
`api.github.com` on 2026-11-24. The top-level token of every workflow is read-only.

## What the owner has to do

1. **Activate the workflows.** Either push with a token that has the `workflow` scope, or commit the files in the
   GitHub web editor:

   ```sh
   cp tools/ci/tests.yml tools/ci/release.yml tools/ci/scorecard.yml .github/workflows/
   ```

   The tree already holds an older `.github/workflows/tests.yml` (three Python versions, no audit); `tools/ci/tests.yml`
   replaces it.
2. **First run of `reference-stack`.** The golden digests were recorded on this project's development machine
   (x86_64, the SIMD extensions listed in `tests/golden/engine_digests.json`). The job forces a bit-for-bit comparison on
   the same numpy and scipy. If a GitHub runner produces other last bits, the job fails on its first run; record the
   file on the runner then (`python tools/audit/golden.py --write --new-reference-stack`) and commit it. Not verified
   here: no GitHub runner was available.
3. **Trusted publishing.** On PyPI, add a trusted publisher for project `mrgym`: owner `KeWang0622`, repository
   `MRGym`, workflow `release.yml`, environment `pypi`. In the repository settings, create the environment `pypi`
   (optionally with a required reviewer). No token is stored anywhere.
4. **Branch protection.** Require the jobs `lint`, `test`, `quality` and `build` on `main`.
5. **Dependabot.** `.github/dependabot.yml` already groups action updates; it moves the pinned commits.

## What was run and what was not

Run on the development machine: everything `make audit` runs (the `quality` and `build` jobs' commands). Not run: any
workflow on GitHub; macOS and Windows; the lowest-pins job; `zizmor` (not installed here); the release workflow and
Scorecard. The YAML files were parsed (`check-yaml`) and nothing more.
