# Character DNA test suite refactor plan

Reduce complete test execution to at most **25% of the matched baseline**, while preserving every existing behavioral contract and improving failure-path coverage. Implementation is underway on `rbf-shape-keys` and the editors submodule's `main` branch. **The runtime target is not yet achieved.**

The implementation separates Python domain tests from Blender tests, leases private scene/DNA fixtures from process-local pristine artifacts, narrows component profiles, and retains the original numerical matrices. The assertion-boundary ledger is `scripts/test_suite_refactor/contracts.json`; exact case and source mappings are in `migration-map.json`. Measurements retain incomplete phase records and compare every original case, covered line and arc. Numerical references, tolerances and existing skips remain unchanged.

The latest complete local measurement is frozen parent `8f337bc` with editors `c6a0574`: **1,512.62 seconds (25m 13s), 25.40% of baseline**. It collects 4,208 cases: 4,142 passed, the same 42 pre-existing Unreal numerical failures remain, and 24 existing cases skip. Every one of the 4,101 original cases is retained, with no unfinished cases, new failures or new skips; 11 original failing cases are repaired. No originally covered line or arc is missing or unmapped. Source hashes and the recorded environment match. This shared-cache diagnostic still exceeds the 1,488.81-second local ceiling by 23.81 seconds and does not replace repeated cold-cache qualification.

| Complete local measurement | Process seconds | Baseline ratio | Cases | Missing covered lines / arcs |
| --- | ---: | ---: | ---: | ---: |
| Original `f5f204b` | 5,955.25 | 100% | 4,101 | Baseline |
| Third candidate `81286ad` | 2,192.53 | 36.82% | 4,176 | 6 / 10 |
| Fourth candidate `bb9b05a` | 1,960.00 | 32.91% | 4,186 | 1 / 6 |
| Fifth candidate `776be85` | 1,830.73 | 30.74% | 4,194 | 0 / 0 |
| Sixth candidate `835d9f8` | 1,648.92 | 27.69% | 4,197 | 0 / 1 |
| Eighth candidate `2b829d1` | 1,560.76 | 26.21% | 4,202 | 0 / 0 |
| Ninth candidate `8f337bc` | 1,512.62 | 25.40% | 4,208 | 0 / 0 |

The original local baseline had 51 failures, 4,024 passes, 24 skips and two setup errors. The first exploratory runs overlapped and had different reporting overhead; their timings are not acceptance evidence. The complete local evidence is in `reports/test-suite-refactor/comparison-v3.json` through `comparison-v9.json`, and each `matched-windows52-candidate-*-r1/runs.json`. The fifth iteration's native/fixture checks passed all 127 cases on Blender 4.5. Blender 5.2 passed every focused test body; its fixture teardown guard was corrected and the affected 16-case group passed again. Both Python versions passed all 2,286 pure contracts.

[Qualification run 40996605128](https://github.com/poly-hammer/character-dna-addon/actions/runs/40996605128) measures the earlier published candidate `0695efe`. Completed cold pairs so far are:

| Lane / repetition | Original seconds | Earlier candidate seconds | Ratio |
| --- | ---: | ---: | ---: |
| macOS 4.5 / 2 | 5,985.79 | 3,329.01 | 55.62% |
| Ubuntu 5.2 / 2 | 7,251.25 | 3,808.99 | 52.53% |
| Windows 4.5 / 1 | 7,963.26 | 4,460.47 | 56.01% |

These reports retain all original cases with no new failures or skips, but fail the runtime and older coverage gates. They establish complete platform baselines; they do not qualify later local revisions. Downloaded artifacts are under `reports/test-suite-refactor/ci-0695-*/`. Publication of the local editor revision is awaiting explicit approval after automatic approval review rejected the push, so the later revisions are currently validated locally.

The eighth candidate prunes UV fallback candidates through the existing spatial grid, avoids ownership inspection for native mute no-ops, and avoids leaving/re-entering Pose Mode during numerical comparisons when the correct rig is already active. The original dense UV scan is retained as an independent test oracle, including boundary, tie, invalid and extreme-coordinate queries. The focused numerical/calibration/ownership runs each retained the same 42 baseline failures and passed 571 cases on Blender 5.2 and 4.5. All six final UV contracts passed again on both versions. These are correctness checks, not complete-suite runtime acceptance.

The ninth candidate reuses field schemas only during one RNA serialization, excludes preview-only flags in that traversal, uses scalar integer clamps for UV grid cells, indexes Blender vertex groups for each weight-export operation, and preallocates the calibration skin table before writing its rows. Independent original serializer and grid-clamp oracles plus exact compact native head/body weight rows guard these changes; the original full export/calibration oracles remain. Three key-identity/rollback cases now use compact native geometry with a real long-named head channel. The full 42-target recovery/import case and combined-neckline portability case retain their complete original geometry. The compact asset still matches every native output at all LODs. The complete ninth run retains all original coverage but remains 23.81 seconds above the local runtime ceiling.

The ninth candidate's final compact-fixture/native-contract group passed all 54 cases on Blender 5.2 and 4.5. The broader exporter/calibrator/Unreal validation retained the same 42 baseline failures on both versions, with 690 passing cases and nine existing skips after correcting a new test's vector-property access; that corrected test also passed independently on 5.2. Both Python-only layers passed all 2,287 contracts without importing Blender.

The tenth candidate narrows only named-bone Swing/Twist contracts to all swing/twist outputs, native input bones, observed state bones and their ancestors. Full-skeleton pose comparisons, complete knee deformation, offset/restoration matrices and undo keep their original scene bones. Combined-file transactions still compare untouched head data, surface identity and coordinates. A multi-character native ownership test retains its fresh second import and complete skeleton but omits its unused mesh; invalid-buffer/index tests use the existing state profile. Complete native DNA and evaluator output arrays remain unchanged. All 80 focused cases pass on Blender 5.2 (115.46 seconds) and 4.5 (117.73 seconds). Complete runtime and exact coverage qualification remains outstanding.

## Baseline and success criteria

The sixth candidate adds operation-local RBF routing indices, avoids repeated linear Blender vertex-group searches during skin-weight import, and avoids recursive calls for scalar graph leaves. Its focused editor/graph/visibility checks passed on Blender 5.2 (111 passed, one existing skip) and 4.5 (103 passed, nine existing skips). After the additional skin-weight and RBF group-index changes, all 31 adapter/component contracts passed on each version, with one existing skip. These include exact full head/body skin-weight comparisons, preservation of unrelated groups, safe group replacement, and every native RBF pose compared with uncached reads. Both Python versions also passed all 2,286 pure contracts.

The seventh candidate uses C-level scalar checks in graph comparisons, retains exact JSON digest bytes, and restores full reference/topology/second-character source rigs through owned leases. The second-character import itself remains fresh. Original and private reference DNA bytes are protected. Component-routing checks retain complete native DNA and solver data with observed scene bones, while full numerical matrices remain in their integration suites. The focused graph/reference/import group passed 114 tests on Blender 5.2; the expanded Blender 4.5 group passed 124 tests with one existing skip. Both Python-only runs passed 2,287 contracts. Complete timing and coverage qualification remains outstanding.

The audit targets parent revision `f5f204b34673b0717404741b964f8de885ba9990` on `rbf-shape-keys`. [GitHub Actions run 40935964822](https://github.com/poly-hammer/character-dna-addon/actions/runs/40935964822) collected **4,101 cases**. All five test jobs reached the 90-minute execution limit. That is a censored baseline: elapsed time before cancellation does not tell us the complete suite duration, and the remaining test cases need not have average cost.

CI already runs pytest in **one Python process with bpy per platform job**. Most setup duplication comes from repeated scene/DNA imports inside that process. A few tests deliberately create subprocesses for behaviors that require process boundaries. Eliminating a supposed Blender launch per test would address the wrong problem.

Measure two separate outcomes:

1. **Work performed:** sum of complete pytest process wall times across the same supported matrix, including fixture construction, serialization, teardown, and coverage/report generation. Require `new / baseline <= 0.25`.
2. **Developer wait:** complete matrix critical-path time, with queue delays reported separately. Require the same ratio. Sharding may improve this metric, but cannot count as a reduction in work performed.

Use matching source/contracts, assets, bindings, OS runner class, Blender patch version, Python, coverage configuration, dependency lock, CI sampling policy, and cache policy. Run at least three complete baseline/candidate pairs per lane, alternate their order, report medians and ranges, and investigate regressions in the slowest runs. Measure both cold setup and warmed reuse. A prebuilt cache must not make its construction cost disappear from the comparison.

The initial engineering ceiling is **22 minutes per platform lane**, conservatively below 25% of the already-observed 90-minute job limit. The final gate must use complete matched baselines. Keep the five existing lanes: Windows/macOS on Blender 4.5 and 5.2, plus Linux 5.2. Adding Linux 4.5 is a separate compatibility task.

The current run also contains 49 failures on Windows/Linux and 50 on macOS before cancellation. Record and resolve these independently of the speed migration: shape-lock fake fixtures, Unreal finger-pose parity, profiling-package expectations, body shape preview, deleted-mesh export, and a macOS combined-neckline backup case. Preserve their reproduction cases and tolerances. Do not obtain a faster green run by skipping, xfail-marking, or weakening them.

## Audit findings

Tracked sources contain **105 test modules and 772 test functions**. Static inspection found 98 fixture definitions: 46 function, 14 module, and 38 session scoped. There are 38 DNA import call sites, 71 scene-reset call sites, 64 DNA-read call sites, and six child-process call sites. These are source locations, not runtime invocation counts.

Collection with `RUNNING_CI=1` confirms these important fixture consumers:

| Fixture | Current scope | Collected cases using it | Refactor direction |
| --- | --- | ---: | --- |
| `test_merge_split.character` | Function | 92 | Split body/head/both/all-LOD profiles; snapshots for eligible scene tests; retain independent files for writeback tests. |
| `test_rbf_edits.fresh_rbf_test_scene` | Function | 47 | Reuse pristine body artifacts; own mutable state per case; keep real native serialization/evaluation. |
| `fixtures.scene.load_head_only_dna` | Function | 37 | Separate bones/face-board needs from geometry/material needs; use owned snapshots where valid. |
| `test_swing_twist_editor.character` | Function | 34 | Separate body and combined-document profiles; preserve full RBF plus swing/twist preview integration. |
| `test_editor_runtime_lifecycle.character` | Function | 20 | Fast base-lifecycle contracts plus real per-editor runtime integration on restored profiles. |
| `test_coordinate_migration.legacy_body` | Function | 17 | Explicit legacy profiles; preserve real migration, basis, and animation assertions. |
| `test_component_editing.component_reader` | Function | 12 | Keep native readers owned; smaller layers only where a measured case benefits. |
| `fixtures.scene.load_body_dna` | Session | 436 | Immutable source artifacts plus explicit scene ownership; avoid assuming another module left its scene intact. |
| `fixtures.scene.load_head_dna` | Session | 276 | Same ownership correction; preserve existing amortization of expensive exports. |

Session scope is not equivalent to isolation. `test_body_rig_logic_unreal.unreal_body` currently imports again the Maya body on teardown because a session fixture otherwise points at the wrong global Blender scene. Other tests call `read_homefile` independently. Increasing fixture scopes without defining scene ownership would amplify this problem.

CI timestamps identify editor lifecycle, merge/split, operation backups, RBF shape creation, and unapplied-pose editing as major costs. On the Windows 5.2 prefix, these include approximately 13.1 minutes in editor lifecycle, 9.5 in operation backups, 6.6 in RBF shape tests, and 6.4 in unapplied-pose tests. macOS 5.2 had already spent approximately 29.2 minutes in merge/split. These are adjacent-log-event estimates, not precise pytest phase durations or completed module totals.

The **1,467 shape-name classification cases** and other naming matrices are inexpensive. Keep them exhaustive. The **431 Unreal body cases**, including provenance and every finger pose in runtime/editor modes, take roughly two minutes on several lanes and provide unusually valuable numerical coverage. Do not thin that matrix to meet the time target.

## Measured pilots

Initial measurements use Windows, Python 3.13.11, bpy 5.2.0 LTS, coverage 7.13.5, branch coverage, and the same source revision. Times below are pytest summary times, including collection and fixture work. These initial comparisons are single pairs unless otherwise noted; repeated cross-platform measurements remain required.

| Pilot | Baseline | Candidate | Result and decision |
| --- | ---: | ---: | --- |
| Three component-selection tests | 59.93 s | 6.48 s | **10.8% of baseline**, all three pass unchanged. Real RNA, callbacks, triangle meshes, and armature modifiers replace DNA imports. Prioritize this pattern. |
| Six editor failure/merge tests | 165.39 s | 81.89 s | **49.5%**, all six pass unchanged; includes two cold imports and snapshot construction. Useful infrastructure, insufficient alone. |
| Three body RBF shape workflows | 94.32 s | 35.36 s | **37.5%**, all three pass unchanged. Keep complete body DNA, native evaluation, commit/revert, file publication, and weight/mirroring assertions; omit the unused head. |
| Same RBF workflows with body snapshots | 94.32 s | 24.79 s | **26.3%** including collection/reporting; setup/call/teardown totals fall from 86.47 s to 18.02 s (**20.8%**). All three pass unchanged, including commit followed by restored pristine state. |
| Four behavior-only native cases | 8.74 s | 11.15 s | Both runs: three pass, one existing single-solver skip. Smaller DNA reads did **not** improve this test group; exclude this from projected savings. |
| Twelve new lifecycle contracts | — | 0.048 s of call time | Actual `Editor.enter` with spec-constrained suspend/resume and surface-session collaborators. All pass; all twelve detect an intentionally omitted resume call. Retain real binding tests alongside them. |

The native reader microbenchmark alternated full/smaller reads five times. Median head read improved from 102 ms to 44 ms; body from 13.8 ms to 1.12 ms. It compared finite joint, blend-shape, and animated-map outputs over 24 head and 12 body evaluations at all LODs, plus setup/count equivalence. The lack of end-to-end improvement demonstrates why microbenchmark ratios must not be applied to whole modules.

Snapshot iteration found two correctness traps. Reopening a blend alone left Python authoring readers absent. Calling `initialize()` without preserving bindings then removed native runtime bindings. The passing version initializes under `controller.preserve_bindings()` and checks active runtime, binding integrity, and reader availability. Never count the failed variants as speedups.

Coverage totals also need scrutiny: the passing six-case snapshot pilot covered 10,194 lines and 1,706 branches versus 10,187 and 1,703 initially, but still missed ten previously hit lines in runtime evaluation/context-buffer setup. Aggregate growth does not prove coverage preservation. Dedicated evaluation tests must retain those paths; compare exact covered lines and arcs across the complete migrated suite.

Reproduction scripts and limitations: [benchmark pilot guide](../../scripts/test_suite_refactor/README.md). [Recorded phase timings](../../scripts/test_suite_refactor/measurements.json) use the instrumentation plugin's wider session clock, so they differ slightly from the pytest summary times above. Raw logs, collection graphs, timing JSON, native coverage data, and XML are in `reports/test-suite-refactor/` locally. The snapshot prototypes are intentionally restricted to selected cases and are not production fixtures.

## Target suite structure

| Layer | Fixture/data policy | Assertions that belong here |
| --- | --- | --- |
| Python domain tests | Frozen records, small arrays, protocol fakes, real temporary files when testing filesystem behavior. No global bpy replacement. | Routing, validation, names, dependency graphs, ownership decisions, UI presentation decisions, backup retention policy, lifecycle transitions, malformed input. |
| Native DNA tests | Real SDK and RigLogic; owned readers/writers/states; explicit required layers; immutable parsed reference data. No scene import. | Numerical RBF/swing/twist output, coordinate conversion, serialization, native schema compatibility, preservation of untouched behavior and geometry. |
| Small Blender tests | Real registered RNA, operators, mesh/key/armature data, and dependency graph on minimal scenes. | Selection, callbacks, collection visibility, locking, actual key ownership, geometry scatter/reorder, Blender API boundaries. |
| Character integration | Complete real component profiles; restored scene and file state for every mutable case. | Real mesh deformation, full-pose preview, native bindings, commit/revert, combined documents, scene merge, all-LOD bridges, animation, undo and reload. |
| Process integration | Fresh process only when process lifetime is part of the contract. | Saved-file portability, native namespace recovery, worker protocol/startup, reload, linked/reference behavior and interpreter shutdown. |

Move Blender-heavy imports and autouse addon registration out of the root shared bootstrap into the Blender test subtree. A Python-only subtree must be collectable without bpy or native bindings. Today, importing `character_dna` and the root conftest pulls those dependencies in; extract genuinely independent production modules or introduce narrow adapters during implementation. Do not fake the whole `bpy` module or execute source snippets as the production test architecture.

Keep pure tests on both supported Python versions. Keep SDK, Blender, file portability, and native ownership tests on the existing platform/version matrix. Moving an unchanged expensive suite to a nightly job does not count toward the fourfold reduction. A scheduled extended corpus may add coverage beyond the existing CI set after that set meets the target.

## Fixture design

Define an explicit `RigProfile` describing component set, source DNA/basis, LODs, geometry, shape keys, materials, face board, document mode, and native runtime requirements. Initial profiles should cover body LOD0, head bones/face board, head LOD0, both components LOD0, all imported LODs, combined documents, and legacy coordinates. Build only profiles actually requested in a worker.

Store immutable source bytes, reference JSON/arrays, and pristine blend artifacts at session scope. Give each test a function-scoped scene lease and independent writable DNA/output directory. Do not share live RNA references or mutable native handles across scene reloads. Read-only metadata can be cached; changed files, canonical-coordinate writers, and ownership tests need fresh handles.

A snapshot key includes source hashes, addon/editor revisions, binding build, Blender version, Python ABI, platform, profile options, and a fixture schema version. Do not share blends or native caches between Blender versions or architectures. Prefer per-job caches first; persistent CI caching is a separate optimization after correctness is proven.

The production restore sequence must:

1. Finish or abort active edit sessions and release native ownership through existing lifecycle helpers.
2. Restore pristine DNA into the test's own directory; bind all component, combined-document, native-carrier, backup, and output paths to that directory.
3. Reopen the pristine scene and reacquire every RNA object by stable identity. Initialize required authoring readers while preserving valid native bindings.
4. Restore frame, pose, selection, mode, visibility, auto-key, undo preferences, component selection, and editor flags. Reset editor/runtime caches and pending callbacks using their owners' cleanup APIs.
5. Verify profile fingerprints, source-file hashes, mesh/key/driver counts, no suspended editor, no binding issues, and correct active component before yielding.
6. On exit, detect leaked IDs, handlers/timers, state, and writes outside the sandbox. Run a following canary test to prove restored state rather than trusting the cleanup implementation.

The pilot's canonical per-session DNA paths are insufficient for backup/path-ownership tests. Path remapping, changed-file invalidation, output cleanup, same-basename components, native carrier descriptors, and combined partition paths must pass contract tests before those modules adopt snapshots. Importer-specific tests must continue importing fresh DNA.

## Migration by test family

| Priority | Existing family | Proposed change | Real coverage that must remain |
| --- | --- | --- | --- |
| 1 | `test_output_component_selection`, editor empty-state UI, selection/lock tests | Minimal registered properties and tiny Blender objects; immutable fake reader metadata where needed. | Head/body export operator round trips, actual component refresh, native suspension lock, Free/Pro visibility. |
| 1 | `test_editor_runtime_lifecycle` and shared editor helpers | Move transition/error permutations to spec-constrained contracts. Restore profiles for concrete editor callbacks. Add tracking/overlay/cleanup failure cases. | Each editor's successful commit/revert and real failed-entry recovery, live jaw/body output, cache-loss lock restoration, no legacy migration state. |
| 2 | `rbf_editor/test_shape_driven`, `test_unapplied_pose_edits`, `test_rbf_edits` | Body-only or head-only profiles where the assertion permits; snapshots; native scalar/membership math separate from Blender UI. | Native coefficient round trip, unchanged joint outputs, shape application/mirroring, write failure/retry, undo, paired/combined rigs and both commit choices. |
| 2 | `test_swing_twist_editor` | Frozen setup records for validator/axis/weight permutations; real native evaluator for full output; restored body and combined profiles for sessions. | Full pose including RBF companions, live blend/axis preview, rest reset, visibility restoration, shared-axis rules, commit/revert and unmodified geometry. |
| 3 | `test_merge_split`, LOD/visibility/source/bridge suites | Small real meshes with independent coordinate/ownership oracles for topology, mapping, seam and failure permutations; reuse full-character profiles for transport. | SCENE and DNA modes, head/body seams, all LODs, shape/driver ownership, transformed/posed geometry, rollback, file reload and combined portability. |
| 3 | `test_dna_operation_backups`, restore visibility | Real temporary file transactions with small content for atomicity; typed collaborators for policy; manifest/retention tests without rigs. | Pre/post actual-DNA backup, overwrite restoration, partial failure, scene mode and visibility restoration, same-basename components, disk portability. |
| 4 | Exporter/calibrator and DNA JSON fixtures | Preserve existing one-time exports; share immutable expected arrays; vectorize full-array comparisons without reducing elements or changing tolerances. | Complete geometry/normal/skin/behavior preservation, deleted mesh mapping, every currently checked LOD, coupled head/body seam export. |
| 4 | RigLogic, coordinates, native ownership, animation and references | Keep efficient matrices; isolate dataset ownership; use targeted component profiles; amortize reader/reference parsing. | Every Unreal finger pose, both runtime/editor paths, provenance hashes, native callbacks, units/bases, animation, reference protections and cross-process recovery. |

For every moved test, record the original node/parameter IDs, assertions, dependency boundary, replacement cases, retained integration anchor, and timing/coverage delta. A test may be split across layers, but no contract may disappear. Avoid a single giant scenario that hides later assertions after the first failure; preserve diagnosable case IDs and per-case restoration.

## Coverage safeguards

Use the same coverage source tree, exclusions, and branch settings. Merge coverage from all layers before comparing exact covered lines/arcs; normalize paths across platforms and include the editor submodule. Total percentages alone can hide lost behavior or an enlarged denominator. Add a reviewed mapping when production code is extracted and line numbers change.

Require typed/spec-constrained mocks at external boundaries, with independent native/Blender contract tests for their assumptions. Mock a write failure or runtime collaborator, not the evaluator/result that the test claims to validate. Keep golden numerical references independent of current outputs, preserve all tolerances, and do not regenerate fixtures just to make refactored tests pass.

Exercise deliberate faults: missing resume, wrong component destination, omitted rollback, reversed axis/unit conversion, dropped RBF contribution, stale shape coefficient, lost visibility state, duplicate publication, and stale reader reuse. Each should fail a targeted cheap test where applicable and a retained integration test for its adapter. The missing-resume pilot demonstrates this for one fault only.

Add fixture isolation tests that mutate pose, selection, visibility, shapes, files, handlers, editor flags and native buffers, then acquire a new lease and compare the pristine fingerprint. Run affected groups in normal, reverse, repeated, and seeded randomized order. Never share a live Blender scene between parallel threads. Detect leaked native handles and monotonic resident-memory growth over repeated leases.

## Runtime budget

The following is an **implementation budget, not a measured prediction**. It intentionally reserves room for cache construction and reporting rather than multiplying the fastest microbenchmark by the whole suite.

| Work per platform lane | Maximum allocation |
| --- | ---: |
| Domain contracts and small filesystem checks | 1 minute |
| Native DNA and exhaustive numerical matrices | 3 minutes |
| Minimal Blender adapters and restored component editing | 6 minutes |
| Retained full-character and process integration | 8 minutes |
| Cold profile construction, teardown, coverage/reporting and variance allowance | 4 minutes |
| **Complete lane** | **22 minutes** |

Track measured time by layer/profile after each migration. Model total time from measured case costs: `new total = fixed cost + sum(old bucket cost × measured retained ratio) + new setup cost`. Do not double-count overlapping savings from body-only imports and snapshots. The initial pilots show roughly 11% retained time for the selection group, 26% for RBF with body snapshots, and 50% for paired-character cold snapshots; a blanket snapshot rollout cannot justify a 25% whole-suite claim. Only the combined RBF measurement supports the combined optimization; multiplying its separate microbenchmark ratios would overstate the saving.

If a bucket exceeds budget, profile its remaining setup/call/teardown costs, split another repeated orchestration/adapter contract out of full-character setup, and benchmark again. Keep numerical and process contracts intact. If retained mandatory integration alone exceeds the ceiling, report that measured floor and revise the fixture/production boundary before expanding the migration; do not hide it with scheduling or extra machines.

## Implementation sequence and gates

1. **Establish a complete baseline and contract ledger.** Save collection IDs, fixture graph, import/process counters, phase durations, memory, exact coverage arcs, versions and source hashes. Complete the currently timed-out baseline using an isolated benchmark run with enough time. Keep correctness fixes separate and establish a new baseline for changed behavior. Exit: complete matched measurements and a disposition for every existing case.
2. **Introduce test layers and instrumentation.** Move autouse registration into the Blender subtree; make pure collection independent; retain explicit native/process bootstraps. Add markers and report merging without dropping checks. Exit: same collected contracts and full-suite coverage, with explainable failures unchanged.
3. **Build and validate owned fixtures.** Implement profile factories, immutable artifacts, per-test files, safe native restoration and leak checks. Start with the proven RNA pattern and the narrowly validated component/snapshot profiles. Exit: order-independent tests, source-byte protection, valid runtime and bounded memory.
4. **Migrate cheap UI and shared orchestration.** Port the pilot patterns, add failure permutations, preserve concrete editor/native integration anchors. Exit: same assertions, missing-resume and rollback mutations detected, measured bucket reduction.
5. **Migrate RBF, swing/twist, merge and backups.** Work one family at a time, retaining all combined-document, all-LOD, pose and portability contracts. Run old and new suites side by side during migration; remove redundancy only after the contract/coverage comparison passes. Exit: per-family ratios and coverage meet the ledger, with no new skips.
6. **Optimize remaining native and export costs.** Use measured profiles to select array/reference caching and fixture boundaries. Keep the reader-layer optimization optional because its first whole-test pilot regressed. Exit: all lane budgets accounted for, not estimated from parameter counts.
7. **Validate the complete matrix and enforce the target.** Run at least three matched repetitions per lane, cold and warm, with coverage and actual process totals. Require <=25% for work and critical path, no lost contracts/arcs, no regression in failures, and no leaked state. Only then add a performance regression gate and retire the old execution path.

During iteration, preserve incremental per-case timing/result records so cancellation does not erase diagnostic evidence. In the reusable workflow, keep pytest's true exit code, publish existing diagnostics on failure, and treat absent reports as absent rather than uploading empty JUnit files. Pin the workflow revision during performance comparisons; a floating `@main` can otherwise change the benchmark conditions.

The first implementation change should be instrumentation, baseline capture, and the fixture contract—not a broad rewrite of every test. The next changes should apply the measured minimal-RNA and component-specific patterns. Full-suite adoption is gated by the coverage ledger and complete timings above.
