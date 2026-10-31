# Test suite refactor measurements

The implemented suite uses `tests/unit` for independent Python domain tests and `tests/blender` for real Blender/native tests. `tests/fixtures/rigs.py` owns pristine, process-local rig artifacts and independent writable DNA for every scene lease. The complete CI corpus remains selected by `RUNNING_CI=1`; no existing numerical parameters or tolerances are removed.

## Run and measure the implemented suite

```powershell
$env:RUNNING_CI='1'
.venv/Scripts/python.exe -m pytest tests/unit --require-no-blender -o addopts=''
.venv/Scripts/python.exe -m pytest tests --test-run-report=reports/full-suite.json
.venv/Scripts/python.exe -m pytest tests/blender/test_rig_fixtures.py tests/blender/test_editor_runtime_lifecycle.py --test-order-seed=284 -o addopts=''
```

`--test-run-report` writes an append-only `.jsonl` event stream plus a final JSON summary. Interrupted runs retain their completed phases and fixture graph. `--test-profile=reports/profile.prof` saves an optional cProfile report before Blender's forced shutdown. Profiling is diagnostic overhead and must be disabled for acceptance timing.

Create frozen source directories before measuring, and run baseline/candidate **sequentially** on the same idle machine. A source snapshot includes the editor submodule, installed native bindings, and test inputs. The original baseline must come from the pre-refactor revision, not a current checkout with different fixture flags.

```powershell
.venv/Scripts/python.exe scripts/test_suite_refactor/freeze_suite.py reports/test-suite-refactor/candidate-frozen
.venv/Scripts/python.exe scripts/test_suite_refactor/run_benchmarks.py --baseline-root reports/test-suite-refactor/baseline-source --candidate-root reports/test-suite-refactor/candidate-frozen --repeat 3 --label acceptance-windows52
```

The runner alternates order, retains pytest's exit code, enables branch coverage, records process time separately from fingerprint computation, and rejects source changes during a run. Without `--baseline-root`, it compares the current fresh-import path (`--fresh-rigs`) against current snapshot restoration; that is a **fixture benchmark**, not the whole-refactor baseline. Do not run other Blender jobs concurrently with acceptance measurements. Build every profile cold in each process; filesystem warming is reported through alternating repetitions, not persistent blend reuse.

```powershell
.venv/Scripts/python.exe scripts/test_suite_refactor/compare_runs.py reports/test-suite-refactor/acceptance-windows52/1-baseline.json reports/test-suite-refactor/acceptance-windows52/1-candidate.json --baseline-root reports/test-suite-refactor/baseline-source --candidate-root reports/test-suite-refactor/candidate-frozen --baseline-coverage reports/test-suite-refactor/acceptance-windows52/.coverage.1-baseline --candidate-coverage reports/test-suite-refactor/acceptance-windows52/.coverage.1-candidate --output reports/test-suite-refactor/comparison-windows52.json
```

The comparison uses `migration-map.json` to retain original case identities, including the exhaustive naming matrices moved to the Python layer. It rejects missing cases, incomplete teardown, new skips/failures, mismatched environments, ratios above 25%, and lost or unmapped covered lines/arcs. Unmapped code is a review requirement, never an automatic waiver. Process totals from `runs.json` must also meet the target, including collection, coverage publication and shutdown. Existing failures remain visible and are compared separately from regressions.

`contracts.json` records fixture ownership and each assertion moved across a mock boundary, together with its retained integration coverage. The numerical pose/reference matrices are unchanged. Validate fault sensitivity with:

```powershell
.venv/Scripts/python.exe scripts/test_suite_refactor/run_mutations.py --output reports/test-suite-refactor/mutations-52
reports/ci45-venv/Scripts/python.exe scripts/test_suite_refactor/run_mutations.py --output reports/test-suite-refactor/mutations-45
```

The mutation runner first requires the unmodified control cases to pass. Separate subprocesses then inject missing resume, rollback and abort cleanup; wrong component routing; reversed preview rotation and incorrect angle units; omitted RBF and shape contributions; lost collection visibility; duplicate document publication; and stale document views. Every selected test must finish setup and teardown, execute the mutated production function, and produce the expected detection count. A surviving fault or a harness/setup failure rejects the diagnostic. These deliberately failing subprocesses are not ordinary suite failures. Use repeated `--fault` options to select particular checks.

Faults are injected in memory only during test bodies, including through previously imported aliases. Fixtures are constructed with unmodified production code, which is restored before teardown. Reports retain the executed function's source hash, invocation count and exception. Output directories must be new so earlier evidence is preserved.

For repeated lease ownership and memory observations, select exactly the two existing canaries below. This opt-in diagnostic does not add repetition to normal CI:

```powershell
$env:RUNNING_CI='1'
.venv/Scripts/python.exe -m pytest tests/blender/test_rig_fixtures.py::test_rig_leases_restore_pristine_state_and_own_paths tests/blender/test_rig_fixtures.py::test_failed_edit_does_not_leave_suspended_runtime_or_editor_cache -p scripts.test_suite_refactor.lifetime_plugin --lease-repetitions=10 --lease-evidence=reports/test-suite-refactor/lease-lifetime52.json -o addopts='' -o log_cli=false -q
```

Use the Blender 4.5 interpreter for a separate run and output path. Each repetition acquires new private paths, checks pristine pose/files/visibility or failed-session recovery, and releases native ownership through the existing canary assertions. Blender ID and handler counts must remain stable. The append-only report retains memory samples, warm ranges and trends after two warm-up repetitions. Current resident memory is available on Windows; a peak-only sample on another platform cannot establish released-memory stability. Review the samples for sustained growth rather than treating allocator caching or peak RSS as a leak proof.

## Historical planning pilots

The scripts below produced the measurements in the [approved plan](../../docs/plans/test-suite-refactor.md). They are preserved for provenance and run only against the pre-refactor checkout. `run_pilots.py` deliberately rejects the reorganized suite so it cannot mislabel current code as the original baseline. Normal pytest discovery does not collect this directory.

Run from the repository root using the already configured development Python environment:

```powershell
.venv/Scripts/python.exe scripts/test_suite_refactor/run_pilots.py --group rna --repeat 3
.venv/Scripts/python.exe scripts/test_suite_refactor/run_pilots.py --group rbf --repeat 3
.venv/Scripts/python.exe scripts/test_suite_refactor/run_pilots.py --group rbf-cached --repeat 3
.venv/Scripts/python.exe scripts/test_suite_refactor/run_pilots.py --group snapshots --repeat 3
.venv/Scripts/python.exe scripts/test_suite_refactor/run_pilots.py --group native --repeat 3
.venv/Scripts/python.exe scripts/test_suite_refactor/run_pilots.py --group contracts
```

`--group all` runs every group. Each group runs sequentially with branch coverage, separate processes, `RUNNING_CI=1`, unique reports, and alternating baseline/candidate order across repetitions. No GPU, network, or live Blender UI is required. The configured native bindings and checked-out editor submodule are required, as with the existing suite. The runner invokes the existing test bootstrap, which may install the local test bindings.

The groups have different purposes:

| Group | Existing cases | Experiment |
| --- | ---: | --- |
| `rna` | 3 | Real registered properties, callbacks, triangle meshes, and armature modifiers replace full DNA imports for selection-only assertions. |
| `rbf` | 3 | Full real body DNA, mesh, bones, RigLogic, and disk writeback; omit the unused head for body-only shape tests. |
| `rbf-cached` | 3 | Combine the complete real body rig with pristine snapshot restoration; measure the combined optimization directly. |
| `snapshots` | 6 | Build two real character profiles once, reopen pristine `.blend` files, and initialize authoring readers while preserving native bindings. |
| `native` | 4 | Omit geometry layers in behavior-only native round-trip tests. One existing head case skips for single-solver DNA. This pilot did not demonstrate a suite speedup. |
| `contracts` | 12 new contracts and 2 reader probes | Exercise the actual editor base lifecycle with autospecced collaborators; compare full and smaller native-reader outputs. Also run an intentional in-memory missing-resume mutation. |

The mutation run must exit 1 with all twelve contract calls failing. It never edits the addon. It proves sensitivity to that specific fault, not completeness of the mocks. The new contracts do not replace the real runtime tests.

Outputs are written to the ignored `reports/test-suite-refactor/` directory. The runner records process wall time, command, revision, expected exit code, pytest phases, fixture setup times, runtime versions, and coverage XML. Native reader probes also write five alternating read timings and numerical-equivalence results. Coverage XML is sparse pilot coverage, not full-suite coverage. Preserve native `.coverage.*` data to compare exact arcs; aggregate percentages alone cannot establish equivalence.

`measure_plugin.py` uses pytest fixture internals only as instrumentation for this prototype. Do not install it globally. The fixture substitution plugins reject test names outside their explicit pilot scope.

## Complete-suite comparisons

`run_benchmarks.py` runs the original checkout and the current candidate sequentially
with the same interpreter, branch coverage, assets and append-only timing plugin.
It fingerprints sources before and after each subprocess. The default `--cache-policy cold`
gives **each** run its own temporary solver environment and pip cache, so the first
variant cannot pay provisioning costs on behalf of the second. `--cache-policy shared`
is available for explicitly labelled warm-cache diagnostics; do not mix policies in a pair.
Pristine rig artifact construction remains inside every candidate process's clock.
Both variants discard successful per-test temporary directories. After each cold run,
the runner removes its isolated temporary root so repeated pairs do not accumulate
solver environments and copied DNA. Cleanup time is reported separately from pytest
process time. Use `--keep-temporary` when investigating generated failure files;
timing, JUnit, logs and coverage evidence are retained regardless.

```powershell
.venv/Scripts/python.exe scripts/test_suite_refactor/run_benchmarks.py --baseline-root reports/test-suite-refactor/baseline-source --repeat 3 --label complete-cold
.venv/Scripts/python.exe scripts/test_suite_refactor/summarize_benchmarks.py reports/test-suite-refactor/complete-cold/runs.json
```

The dedicated CI workflow is **manual-only**. It defaults to one matched pair on
Ubuntu 5.2; select another supported platform/version or three repetitions when
needed. Each pair is sequential on one runner, and repetitions alternate which suite
runs first. Workflow-wide concurrency and `max-parallel: 1` permit only one benchmark
job at a time, including at most one Mac job across branches and manual dispatches.
Normal PRs retain the ordinary supported test matrix and never launch this comparison.

```powershell
gh workflow run test-suite-refactor-benchmark.yaml --ref rbf-shape-keys -f 'lane=ubuntu-latest / 5.2' -f repetitions=1
```

A selected-lane run does not establish complete-matrix acceptance. For a deliberately
scheduled full qualification, collect three pairs on each of the five supported lanes
at the same frozen revision, dispatching each lane after the previous run finishes.
Download their artifacts into separate `test-refactor-<os>-<blender>-<repeat>` folders.
`combine_benchmarks.py` still requires all fifteen artifacts, consistent source
fingerprints and versions within each lane, and passing case, coverage and runtime
comparisons. Missing or canceled pairs cannot satisfy it. Manual cancellation retains
completed and partial evidence; this expensive qualification is no longer automatic
on refactor pushes because it competes with other repositories for Mac capacity.
`prepare_ci.py` installs the same bindings before source fingerprinting. It preserves
the full numerical corpus, subprocess contracts and existing failures. A known baseline
failure is reported separately from a new regression; no case becomes an xfail or skip.
The acceptance report rejects incomplete runs, missing cases, new failures/skips,
unmapped coverage, changed sources, differing environments/cache policies and runtime
above 25%. Reports include process time as well as pytest phase time.

Coverage normalization follows the reviewed declarations, identifier renames and
statement mappings in `migration-map.json`. Each changed-statement mapping records
its reason and retained contract. An inserted statement must execute on a path between
the same mapped endpoints; traversal may not pass another original statement.
Import-time sequencing across extracted modules or deferred imports is listed separately
and requires both declarations to remain covered. Function branches and unknown mappings
are never silently waived. An existing assertion failure also cannot hide a new setup
or teardown failure on that case.

Raw tracing can record an exceptional function exit at a statement that has no
source exit branch (for example, a stale RNA access in an originally failing
test). The comparator uses coverage.py's source parser to distinguish those
unwinds from its actual branch possibilities and lists them explicitly under
`non_branch_exception_exits`. The originating statement must still execute in
the candidate. Explicit raises, conditional exits, returns and unmapped
statements remain required; fixing an old crash does not require recreating it.

## Historical snapshot limitations

The snapshot prototypes use a private per-run DNA directory, reset their DNA files before reload, and reacquire the rig instance after every load. They do not give each test its usual DNA paths. Do not use them on path-ownership, backup, combined-document, cross-process, or reference-file tests. A production fixture needs per-test path remapping, native carrier path verification, output cleanup, data fingerprints, and state-leak checks before expansion.

Loading a `.blend` alone was insufficient. Plain `instance.initialize()` then removed native bindings. The passing version uses `controller.preserve_bindings()` around initialization and explicitly checks both authoring readers, binding issues, and active runtime. All fixture setup cost, including the first imports and saves, remains in the reported run.

## Recreate CI timing evidence

Download completed job logs with the GitHub CLI, for example:

```powershell
gh api repos/poly-hammer/character-dna-addon/actions/jobs/123666931227/logs --allow-escape-sequences > reports/test-suite-refactor/ci-windows45.log
.venv/Scripts/python.exe scripts/test_suite_refactor/analyze_ci.py
```

The other jobs in run `40935964822` are Windows 5.2 `123666920336`, macOS 4.5 `123666944996`, macOS 5.2 `123666930266`, and Linux 5.2 `123666947995`. Save them with corresponding `ci-*.log` filenames. Logs contain adjacent timestamped test events; module intervals are approximate and exclude the unfinished final test. They cannot replace complete pytest phase timings.

For the current fixture graph without executing tests:

```powershell
$env:RUNNING_CI='1'
.venv/Scripts/python.exe -m pytest --collect-only -q -o addopts='' -o log_cli=false --test-run-report=reports/test-suite-refactor/collection.json
```

The October 29 measurements and their limitations are recorded in the plan. Do not compare historical September reports without matching revision, coverage settings, case inventory, native bindings, and Blender version.
