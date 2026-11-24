## What changes and why

<!-- One paragraph. Link the issue: Fixes #... -->

## Can this change any reward or any scan data?

- [ ] No. Every stored episode keeps its scan data and its reward. (patch)
- [ ] A reward can change for the same scan data. Stored episodes must be re-scored; the environment gets a new
      `-vN` and the changelog says so. (minor)
- [ ] Scan data can change for the same seed and protocol. Episodes must be re-run; `ENGINE_VERSION`, the reference
      numbers in `tests/test_engine_version.py` and the "Engine" line of the changelog are updated. (major)

## How it was tested

<!-- Commands and their output. Say what was not run. -->

- [ ] `make test`
- [ ] `make lint`
- [ ] `make check` (generated files, results table, `mrgym check` on every id)
- [ ] `mrgym validate`, if the scanner changed (paste the table)
- [ ] `mrgym check <id> --admission`, if an environment or its scoring changed (paste the table)
- [ ] No MRI Agent Bench material, no held-out seed and no episode config in the diff
