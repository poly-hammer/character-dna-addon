# Suite benchmarks

These opt-in tools support the manual `test-suite-refactor-benchmark.yaml`
workflow and the benchmark regression tests in `tests/unit`. They are separate
from the domain tests, Blender tests and fixture implementations. Normal pytest
discovery does not launch benchmark subprocesses.

The original refactor pilots, their timing ledger, the source-freezing helper and
the ad hoc mutation/lifetime diagnostics have been retired. Historical evidence
is described in the [refactor plan](../../docs/plans/test-suite-refactor.md), and
the fixture/assertion ownership ledger is [contracts.json](../contracts.json).

## Run a matched comparison

Run from the repository root using the configured development environment.
Prepare separate, fixed baseline and candidate checkouts with their editor
submodules, test inputs and the same installed native bindings. The baseline for
the original refactor is `f5f204b34673b0717404741b964f8de885ba9990`.

```powershell
python tests/benchmarks/prepare_ci.py --bindings ../character-dna-bindings ../baseline .
python tests/benchmarks/run_benchmarks.py --baseline-root ../baseline --repeat 3 --label complete-cold
python tests/benchmarks/summarize_benchmarks.py reports/test-suite-refactor/complete-cold/runs.json
```

Use `--candidate-root` to measure a separate candidate checkout. Both suites run
sequentially with the same interpreter, `RUNNING_CI=1` and branch coverage.
Repetitions alternate order. Source fingerprints must remain unchanged during
each run; do not edit either checkout or run other Blender jobs while measuring.

Each cold run owns an isolated temporary solver environment and pip cache.
Successful temporary files are removed after the run; `--keep-temporary` retains
them for investigation. Process time and cleanup time are reported separately.
The existing `reports/test-suite-refactor/` output location is retained so saved
reports and CI artifacts keep the same layout. Choose a new label for each run.

Without `--baseline-root`, the runner compares fresh imports with snapshot
restoration in the current suite. That measures fixture performance; it does not
reproduce the original refactor baseline. `--cache-policy shared` is an explicit
warm-cache diagnostic. `--no-coverage` cannot produce an accepted comparison.

## Interpret the reports

`compare_runs.py` loads the adjacent `migration-map.json` to match original case
identities and reviewed source mappings. Comparisons reject missing or unfinished
cases, new failures/skips, differing environments, missing or unmapped covered
lines/arcs, and runtime above the original 25% target. Existing failures stay
visible. `summarize_benchmarks.py` also checks process time, unchanged sources,
cache policies and completion, retaining detailed comparison JSON for each pair.

The mapping and threshold describe the original refactor. The user accepted that
refactor's runtime and ended further timing qualification; these utilities remain
available for explicitly requested comparisons, not as a new acceptance campaign.

## Manual CI workflow

The workflow remains manual-only and runs one selected platform/Blender lane,
with one or three sequential matched pairs. Ordinary PRs use the normal test
matrix. Workflow-wide concurrency and `max-parallel: 1` limit benchmark load.

```powershell
gh workflow run test-suite-refactor-benchmark.yaml -f 'lane=ubuntu-latest / 5.2' -f repetitions=1
```

A selected lane is not complete-matrix qualification. When explicitly needed,
collect three pairs on each of the five supported lanes at the same candidate
revision, preserving the `test-refactor-<os>-<blender>-<repeat>` artifact folders:

```powershell
python tests/benchmarks/combine_benchmarks.py reports/benchmark-artifacts
```

The combiner requires all fifteen artifacts, matching versions/source fingerprints
within each lane, cold caches and passing case, coverage and runtime comparisons.
Missing or canceled pairs cannot satisfy it.
