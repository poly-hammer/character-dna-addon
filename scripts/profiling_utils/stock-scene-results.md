# Stock Blender Real-Scene Results

Update 2026-09-08: the runtime now loads a direct platform `.pyd`/`.so` beside the
SDK bindings; no runtime wheel is required. Input matrix/quaternion work and eye
feedback now execute in C++, with bulk RNA capture in Python. New trace error is
at most 6.38e-6 under the existing 1e-5 tolerance. The older wheel commands and
exact-parity measurements below describe previous artifacts, not the current build.

Update 2026-09-07: the user accepted approximately 3.7x and the opt-in runtime is
now integrated. See [Native Evaluation](../../docs/free-features/native-evaluation.md)
for activation, lifecycle behavior and current qualification limits. The original
five-pair results and implementation checkpoint below remain historical evidence.

Date: 2026-09-06. This is integration evidence, not a completed production release.

## Result

The carrier drives Ada's real head/body pose channels, shape keys and animated
material masks. The OpenRigLogic module performs canonical DNA loading, GUI-to-raw
mapping, raw quaternion overrides, solving and local-channel conversion in C++.
The disposable scene binding still captures inputs in Python using the existing
constraint-aware quaternion helper. Production lifecycle integration is **not
complete**, and the experimental add-on preference has not been enabled.

**Full-scene speedup: 3.7264x. The approximately 7x gate is not met.**

| Check                                    | Result                                                                                                                                                   | Evidence under `reports/profiling/stock/`         |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Real Ada trace                           | Exact head/body pose, evaluated mesh and map parity at 125 animation/backward/subframe samples                                                           | `20260906_029/trace.json`                         |
| Five paired scene trials                 | 6,000 measured samples/backend; Python 40.49095 milliseconds, carrier 10.86588 milliseconds                                                              | `20260906_030/summary.json`                       |
| Paired output artifacts                  | All five saved pose/mesh/map snapshots match exactly; runtime/fixture/module hashes match within pairs                                                   | `20260906_030/summary.json`                       |
| Background Cycles renders, frames 1/57/1 | Maximum pixel error 2.3842e-7; viewport output unchanged                                                                                                 | `20260906_031/render.json`                        |
| Single-variable array-trigger variant    | Exact 125-frame scene trace; one timing trial 9.8913 milliseconds; render pixel error at most 1.1921e-7                                                  | `20260906_032/`, `20260906_033/`, `20260906_035/` |
| Native local-transform edge cases        | 160 exact comparisons with Blender pose matrix application, including non-unit quaternions, signed scale and singular matrices; invalid buffers rejected | `20260906_034/transforms.json`                    |
| Prior SDK/cache/session regressions      | Passed after canonical/combined-mode additions                                                                                                           | `20260906_036/sdk-regression.json`                |

## Measurement

Five counterbalanced pairs used fresh stock Blender 5.2.0 processes, 120 warmup
frames and 1,200 measured changing frames each. No benchmarks/builds ran concurrently
with timing. The frame measurement includes input capture, native calculation,
conversion, driver application and downstream scene evaluation. Rendering was
tested separately and is not included in the frame timing.

| Statistic (milliseconds)          |   Python |  Carrier |
| --------------------------------- | -------: | -------: |
| Mean                              | 40.49095 | 10.86588 |
| Median                            | 39.63495 | 10.35365 |
| p95                               | 46.35750 | 14.20610 |
| p99                               | 50.37490 | 15.81100 |
| Minimum trial mean                | 39.91882 | 10.01640 |
| Maximum trial mean                | 41.46152 | 12.87677 |
| Standard deviation of trial means |  0.58104 |  1.24232 |

Do not replace the aggregate with the fastest trial or extrapolate the earlier
3.87-millisecond synthetic stage into a whole-scene claim. The single array-trigger
trial is exploratory, not a replacement five-pair acceptance result.

Each head/body carrier executed 1,320 times per trial, matching warmup plus timed
frames. A representative trial measured about 1.19 milliseconds total inside both
callbacks, of which about 0.43 was Python input capture. Most time remains outside
the carrier callbacks in Blender's graph/driver/downstream work. This does not
identify a specific Blender subsystem as the bottleneck. Porting the remaining
input capture alone cannot recover the gap to approximately 7x.

## Implementation

- `load_model(path, quaternion, canonical=True)` matches the add-on's left/up/front
  coordinate normalization. Preserved-coordinate SDK-only parity was insufficient
  for real scene compatibility.
- `evaluate_into(..., combined=True)` accepts GUI values followed by raw-control
  slots, maps GUI values natively, then replaces only declared quaternion raw
  slots before calculating. Cache keys include the combined mode.
- `transform_into` converts bound joints to local XYZ pose channels in one native
  batch, preserving head non-leaf Euler overrides and body quaternion semantics.
  Do not normalize body output quaternions: the existing matrix path does not,
  and normalization broke parity.
- Compatibility math remains within `OpenRigLogic/modules/blender`.
  `PROVENANCE.md` records Blender reference sources and GPL obligations. No
  Blender-private functions, headers or structure offsets are linked.
- The optional array-trigger probe drives the last output-array element; consumers
  depend on the same array property. It removes the second epoch variable and
  passes current trace/render checks. Actual simulation baking and multi-version
  qualification of this variant remain pending; it is not the default mode.

## Release Gate

This is **not the final production implementation**. Per the agreed plan, the
missed speed gate is recorded rather than silently accepting a lower target.
The real-scene bindings remain disposable profiling probes. They require the
agreed eye-aim-OFF Ada fixture and preinitialized constant channels, support XYZ
output channels, and are not save/reload/undo-safe production bindings.

Remaining requirements include native input capture/planning, eye aim and dynamic
toggle/LOD handling, automatic lifetime/model invalidation, editor compatibility,
undo/load/delete recovery, UI F12/cancellation, actual native-character simulation
bakes, production multi-character scenes and platform wheels. Earlier synthetic
bake/multi-instance tests do not substitute for these real integration gates.
No `RigInstance` facade migration or preference is represented as complete. No
live DNA, original benchmark fixture or Blender source was modified.

Before production promotion, either accept the measured approximately 3.7x path
as a revised target and finish those gates, or keep approximately 7x as a
requirement and investigate a further reduction in Blender-side application cost.
These results do not prove that all stock-Blender approaches are exhausted.

## Reproduction

Use stock Blender with `--background --factory-startup --python-exit-code 1`.
Supply scripts through `--python` and their arguments after Blender's `--`.
Scene scripts require `--fixture`, `--module-dir` and a new `--output`.

- `stock_scene_probe.py --trace`: full real-scene trace.
- `stock_scene_probe.py --stage frames --backend python|carrier --warmup 120 --frames 1200`:
  one fresh-process timing trial, with a saved evaluated snapshot.
- `stock_scene_report.py --root <paired-report-directory>`: run in the project
  Python environment to validate five paired snapshots/provenance and aggregate.
- `stock_scene_render.py`: actual Cycles pixels and viewport isolation.
- `stock_transform_probe.py --module-dir <module-build> --output <report>`: native
  local-transform compatibility and malformed buffers.
- `--array-trigger` on scene/render probes: experimental single-variable
  array-property dependency mode.
