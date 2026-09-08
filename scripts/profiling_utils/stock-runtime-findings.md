# Stock Blender Integration: First Implementation Milestone

Update 2026-09-08: runtime distribution now uses the existing platform bindings
folders and the shared SDK bindings workflow, not a wheel. Eye feedback,
inheritance reconstruction, input quaternion conversion and GUI gathering math
now run in native C++; Python captures RNA arrays in bulk. The full 125-frame
trace remains within the unchanged 1e-5 tolerance (maximum 6.38e-6). The earlier
wheel/prototype entries below are historical; use the activation guide for current
installation details.

Update 2026-09-07: the approximately 3.7x target was accepted. The opt-in runtime
is integrated into the add-on; see the [activation and qualification guide](../../docs/free-features/native-evaluation.md).
The production timing pair measured 39.5183 milliseconds Python versus 10.7955
milliseconds native. Live character simulation bakes, reload/undo, duplication,
eye aim and UI render completion passed. UI render cancellation remains unverified.
The sections below retain the earlier experimental evidence.

Date: 2026-09-06. This is implementation evidence, not a release qualification.

Latest: [real-scene parity and five-pair performance results](stock-scene-results.md).
Real Ada parity passes, but the measured full-scene speedup is 3.7264x rather
than the approximately 7x target. Production backend promotion remains gated.

## Update: Cached Carrier Drivers

The original scheduling blocker below is now narrowed: a **single carrier solve
with public evaluated-buffer publication and simple-expression output drivers**
passes the current probes. It is the recommended next integration path. The
older per-scalar native-call approach remains unsuitable, and no post-frame-only
solution is being promoted for live simulations.

The carrier driver explicitly depends on its inputs, evaluates a context-owned
native session, and fills its own previously allocated evaluated ID-property output array
using the public buffer protocol. Downstream drivers depend on both its epoch and
their output array element. They remain simple expressions, so neither RigLogic
nor Python output getters run once per scalar channel. Blender still owns the
scalar application nodes and downstream dependency ordering. This is one solve
and one publication per necessary revision, not literally one Blender graph node.

Native `evaluate_into` caches the exact float32 input bytes, LOD and GUI/raw mode.
Models are immutable; a model/configuration change requires a new session. Every
valid request republishes outputs even on a cache hit. Twenty repeated identical
requests per component caused zero additional solves while restoring destination
buffers reset to sentinel values. Invalid destinations/non-finite controls left
both outputs and completed-solve counts unchanged.

### New Evidence

All paths below are under `reports/profiling/stock/`:

| Probe                                              | Result                                                                        | Evidence        |
| -------------------------------------------------- | ----------------------------------------------------------------------------- | --------------- |
| Python RNA getter cache                            | Correct but 89.20 milliseconds at 10,062 outputs                              | `20260906_009/` |
| Public evaluated array, distinct synthetic outputs | 5.8793 milliseconds at 10,062 outputs; original carrier unchanged             | `20260906_010/` |
| Cached carrier GN and cloth disk bakes             | Both match reverse-read oracle caches                                         | `20260906_011/` |
| Native cache API and prior Ada parity suite        | Passed; invalidation, cache hits, republishing and rejection checked          | `20260906_012/` |
| Two Ada heads, all 17,388 driven outputs           | Exact at all tested frames and contexts                                       | `20260906_014/` |
| Cloth portability                                  | Cached carrier passes fresh same-version oracles on 4.5.0 and 5.1.0           | `20260906_015/` |
| Native head solve plus 8,694 identity bone drivers | Three trial means 3.9043, 3.9118, 3.8040 milliseconds                         | `20260906_016/` |
| Upstream driver and COPY_LOCATION constraint       | Exact complete arrays; repeated frames, same-frame edits and view layers pass | `20260906_017/` |
| Same native binary in Blender 5.1.0                | Complete constrained-output test passes exactly                               | `20260906_018/` |
| Eight independently animated rigs                  | Isolated contexts; republishing solves 56 -> 56; one edit 56 -> 57            | `20260906_019/` |
| Old scripted-call path at 8,694 outputs            | One diagnostic trial: 85.9667 milliseconds, no SDK computation                | `20260906_020/` |

The newer stage benchmark averages 3.8734 milliseconds, but is **not a complete
Ada scene evaluation**. SDK outputs feed identity bone channels in a synthetic
hierarchy: real rest-space conversion, head/body coupling, skinning, shape/key
and material application are excluded. Simple dispatch at the same count averaged
1.9014 milliseconds over three trials. The old single-trial comparison executes
sine, while the new path actually solves the head SDK; they are diagnostics with
different work, not a release-quality speedup ratio. Full-scene ~7x remains open.

The full output test exposed default driver F-curve endpoint mapping snapping
values smaller than about 0.0001 to zero. Explicit identity mappings for newly
created binding-owned drivers restored exact parity. No tolerance was weakened.
Original output-carrier arrays remain untouched; Blender's normal driver target
writeback to original IDs is allowed. Detaching an action during manual-edit
tests must preserve upstream drivers, not clear all animation data.

### Compatibility And Private API Risk

| Approach                                    | Current evidence and risk                                                                                                                                                                                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Public buffer carrier + simple drivers      | Working prototype without private linkage or layout assumptions; must still qualify evaluated-copy isolation, dependency ordering and lifecycle on each supported build. Publication inside driver evaluation is not a documented general custom graph-operation API.          |
| `as_pointer()` as identity                  | Used only for graph/carrier identity with a separate lifetime generation; no memory dereference. Address reuse requires generation invalidation on undo/load/remap/delete. Automatic production lifecycle wiring is pending.                                                   |
| Direct private structure reads/writes       | Not needed for current progress. Blender version, architecture, compiler and build options can change field layout even within nominally compatible releases; bad access can silently corrupt data or crash. Must be exact-build qualified and fail closed before access.      |
| Private RNA/dependency-graph function hooks | No stable external Blender C++ ABI or supported plugin registration contract. Missing/hidden symbols, signature changes and scheduler changes can cause import failure, wrong ordering, deadlocks or use-after-free. Matching a symbol name or release number is insufficient. |

The native module imported unchanged into Python 3.13 builds of Blender 5.1.0 and
5.2.0. The 4.5.0 check exercises the Python-only public carrier mechanism, not the
CPython 3.13 module. No claim is made for all patch releases, platforms or render
modes. The experimental preference remains unimplemented/disabled, and older
versions retain the existing backend. Do not deploy offset guesses or runtime
private-function patches based on these results.

Next: bind real head/body local outputs to this scheme, preserve complete input
dependencies, and validate real character geometry, UI rendering, undo/reload and
cancellation before the five-pair full-scene performance gate. The current probes
use disposable fixtures and explicit context-generation resets, not a finished
scene lifecycle controller. No private-symbol experiment is justified yet by the
measured public-path results.

## Source And Runtime

- SDK branch: `5.8+blender5.2`, based on `5.8` commit
  `6e8c15b730280a545b757549e5642fb48f597c6c`, SDK 13.2.5.
- Add-on branch at implementation start: `native-cpp-rig-instance`.
- Stock Blender 5.2.0 LTS, build `fbe6228777e7`, Python 3.13.13.
- Executable SHA256:
  `e27fbfea8564aa645d4463cb0949695fd85562b9de6df9561b06859a1074adf7`.
- New C++ code is isolated in OpenRigLogic `modules/blender`, built only with
  `RL_BUILD_BLENDER_MODULE=ON`. No Blender source changes or private symbol use.
- Existing add-on evaluation, preferences, editor state and packaged bindings are
  unchanged. The module reports `production_ready=False` and is not loaded by the
  add-on. No pointer adapter or wheel release is enabled.

## Results

| Check                                       | Result                                                                                                              | Evidence under `reports/profiling/stock/` |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Pose/key bulk RNA                           | Passed tested geometry cases; pose requires explicit owner tag                                                      | `20260906_001/bulk.json`                  |
| Socket collections                          | Float and mixed float/color roundtrips passed; rendered pixels not tested                                           | `20260906_001/bulk.json`                  |
| GN sequential stepping                      | Handler and driver match all 12 accumulator values                                                                  | `20260906_002/`                           |
| Actual GN disk-cache bake                   | Handler and driver match all 12 frames, read back backwards                                                         | `20260906_003/`                           |
| Native SDK parity                           | 48 raw-control/LOD cases plus three GUI cases passed                                                                | `20260906_007/sdk.json`                   |
| Buffer/session lifetime                     | Eight invalid-input cases per component, eight concurrent independent sessions per model, retained snapshots passed | `20260906_007/sdk.json`                   |
| Actual cloth disk-cache bake, handler       | **Failed**; maximum cached vertex component error 0.1442909241 Blender units                                        | `20260906_005/cloth-handler.json`         |
| Actual cloth disk-cache bake, driver        | Passed, absolute tolerance 1e-5                                                                                     | `20260906_005/cloth-driver.json`          |
| Synthetic registered-C-call driver dispatch | Three trial means: 80.4307, 77.1138, 78.2035 milliseconds                                                           | `20260906_006/driver-native-call-*.json`  |
| Synthetic simple-expression dispatch        | Three trial means: 2.3807, 2.2641, 2.2257 milliseconds                                                              | `20260906_006/driver-simple-*.json`       |
| Fresh Python whole-scene baseline           | 38.7064 milliseconds mean; 41.4821 milliseconds p95; 25.84 evaluated frames/s                                       | `20260906_008/python-baseline.json`       |

SDK parity compares all joint, blend-shape and animated-map outputs against the
unchanged packaged SDK using matching float configuration. All 48 raw-case joint
comparisons were exact. Final reports pin module, binding and DNA hashes. This
does not yet validate Blender rest-space conversion or apply the module to Ada.

The initial sequential simulation trial began at the simulation start frame and
had an initialization-only discrepancy. It is superseded by `20260906_002`, which
starts before the simulation range. Real disk-bake results are separate. Each
variant uses a disposable factory-startup process. The ordered test oracle is not
a proposed runtime pre-bake feature. Existing live DNA and benchmark fixtures
were not overwritten.

## Performance Interpretation

The driver trials use 1,118 independent synthetic bones with nine driven channels
each, 30 warmup frames and 240 measured changing frames per fresh process. Trial
order alternates; timing runs are sequential. Both paths call sine, but the
registered C callable forces Python expression execution. Every output is checked
after timing. The three-trial means are 78.5827 milliseconds versus 2.2902 milliseconds.

This is a dispatch diagnostic, not an optimized Ada workload or a speedup claim.
Actual Ada LOD0 has 5,016 head and 2,436 body variable joint attributes, 782 shape
channels and 82 map channels, with fewer distinct material targets. Those counts
do not imply every channel needs a separate dynamic driver. The diagnostic omits
RigLogic, rest conversion, mesh deformation, materials and other scene work.

The fresh Python baseline reuses the verified animated Ada fixture, 120 warmup and
1,200 measured frames, with the current listener explicitly started. It is one
trial, not the final five-pair comparison. A 7x target against it is about 5.53 milliseconds
per frame. No accelerated full-scene result or viewport gain is claimed.

## Original Milestone Decision

The tested post-frame bulk handler cannot meet the live cloth-baking requirement.
Matching final pose or GN caches is insufficient. The direct per-output native
Python-driver design also does not meet the budget at the tested output count.
Do not ship either as the planned all-context solution.

Keep the portable SDK module and regression probes. Before broad `RigInstance`
migration, investigate supported dependency-ordered output access that preserves
Blender's simple-expression path without thousands of Python expression calls.
That mechanism was not implemented or validated in the first milestone; the
cached-carrier update above supersedes this gate. The measurements did not prove
that every possible stock-Blender integration was infeasible.

Still unimplemented/unverified: compiled Blender I/O and rest-transform conversion,
dirty batching/content-addressed model cache, full driver binding and namespace
lifecycle, constrained multi-character scenes, UI F12/viewport-render/cancel paths,
editor integration, actual accelerated Ada scene parity, pointer experiments,
Linux/macOS packages and the final ~7x A/B gate. No simulation pre-bake workaround,
custom Blender fork or unsafe direct-memory writes have been introduced.

## Reproduction

Build the SDK with `modules/blender/build_windows.ps1`. Use the installed stock
Blender executable with `--background --factory-startup --python-exit-code 1`.
Pass each probe via `--python` and its arguments after `--`:

- `stock_riglogic_probe.py --output <new-report.json>`
- `stock_sdk_probe.py --module-dir <SDK-build>/modules/blender --output <new-report.json>`
- `stock_simulation_probe.py --mode oracle --bake --output <new-oracle.json>`, then
  `--mode handler` and `--mode driver` with `--reference <new-oracle.json>` and unique outputs.
- `stock_cloth_probe.py --mode oracle --output <new-oracle.json>`, then both other
  modes with `--reference <new-oracle.json>`. The handler run is expected to exit 1
  after writing the mismatch report. Do not weaken its assertion.
- `stock_driver_benchmark.py --backend native-call --bones 1118 --frames 240 --warmup 30 --output <new-report.json>`,
  paired with `--backend simple` in separate sequential processes.
- Existing `native_frame_benchmark.py frames --backend python --fixture <Ada-fixture.blend> --output <new-report.json>`
  runs the OFF baseline and does not use the fork-only API on that code path.
