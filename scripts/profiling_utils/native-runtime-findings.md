# Native RigLogic Experiment: Findings

## Conclusion

The native Blender implementation achieved **186.63 evaluated scene frames per
second**, versus **25.39** for the existing Python path: **7.35x faster** on the
agreed Ada fixture. The fastest of five native trial averages was **190.91**
evaluated frames per second. These are sustained trial measurements, not a
universal or mathematically proven maximum.

Interactive material preview improved from **12.19 to 20.97 drawn animation
frames per second**, a **1.72x gain**. Moving evaluation into C++ substantially
reduces scene-evaluation cost, but does not make this material-preview workload
run at the headless evaluation rate. Monitor-presented FPS was not measured.

The experiment, compiled executables, validation tools, raw measurements and
source changes are available locally. It is a working native prototype, not
a production-ready replacement for every add-on/editor workflow.

## Workload And Machine

The user explicitly approved finishing with this verified Ada fixture:

- Ada head and body, LOD0, 120-frame deterministic face-board and body-driver
  animation, shape keys, RBF/twist/swing evaluation, and texture-mask updates.
- Head: 870 DNA joints, 867 mapped joint outputs, 782 blend-shape channels,
  five Key targets, and 82 animated-map channels mapped in order to 41 sockets.
- Body: 342 DNA joints, 251 mapped driven joints, and 176 quaternion raw inputs.
- No separate body control rig, groom or production skin albedo/normal texture
  set. The imported images are topology images and the combined mask atlas.
- Eye aim is off in the timed animation. Its new bounded solve and follow
  switches are verified separately; its extra solves are not free.
- Intel Core Ultra 9 285K, 24 cores; approximately 64 GB RAM; NVIDIA RTX 5090;
  Windows 11 Pro; recorded Balanced power plan. No global power settings changed.
- Material preview: 1920 x 1001 window, 1574 x 881 viewport region, fixed
  orthographic framing, overlays off. Playback target 1000 with no frame dropping.

The fixture hash is
`00929bf4cfc169f92c4a81f64176c77176a8f859f5ef18d366a698130972a59b`.

## Final Measurements

Both primary modes used the same executable and byte-identical fixture in
separate fresh processes. Five pairs were run sequentially with counterbalanced
order, 120 warmup frames and 1,200 measured frames per trial: 6,000 samples per
backend. Timed scope is `scene.frame_set()` plus synchronous dependency-graph
completion. Snapshot reading, imports, setup, JSON writing and screenshots are
outside timing. Detailed stage tracing is disabled.

| Scene metric                              |          Python |        Native |
| ----------------------------------------- | --------------: | ------------: |
| Mean frame time, milliseconds             |         39.3917 |        5.3583 |
| Median, milliseconds                      |         38.9121 |        5.1665 |
| P95, milliseconds                         |         42.6828 |        6.8510 |
| P99, milliseconds                         |         49.7159 |        8.5339 |
| Sustained evaluated frames/second         |         25.3861 |      186.6250 |
| Trial mean range, milliseconds            | 38.9251-39.7324 | 5.2381-5.5283 |
| 95% interval for trial mean, milliseconds | 38.9683-39.8151 | 5.2035-5.5132 |

The intervals use five independent trial means and a Student t interval; they
do not treat correlated frame samples as independent trials. Speedup is the
ratio of mean frame times, not an average of reciprocal frame rates. All paired
final snapshots matched within `5.96046448e-7`.

An independently compiled, untouched Blender source baseline measured
**40.2614 milliseconds** per Python frame, or **24.84** evaluated frames per
second, in one corroborating trial. Its final outputs also matched the native
candidate within `5.96046448e-7`. It is not the denominator of the primary
same-binary comparison.

| Material-preview metric                 |          Python |          Native |
| --------------------------------------- | --------------: | --------------: |
| Aggregate drawn animation frames/second |         12.1930 |         20.9696 |
| Three-trial range                       | 12.0968-12.2422 | 20.8751-21.0551 |
| Observed animation frames               |           2,194 |           3,774 |
| Observed duration, seconds              |         179.939 |         179.975 |

Each viewport trial has 10 seconds warmup and 60 seconds observation. A
read-only `POST_PIXEL` observer records completed draws with changed animation
frame IDs while Blender drives playback. This is **not PresentMon/ETW capture**,
does not measure monitor presentation, and is not a renderer-only GPU timing. The
observer overhead is present in both modes. Saved screenshots were inspected
for nonblank output and consistent framing, not used as same-frame parity images.

## Where The Time Goes

An earlier opt-in diagnostic, kept separate from headline runs, measured mean
head input/solve/joint-apply costs of 0.0189 / 0.3322 / 0.1615 milliseconds and
body costs of 0.0311 / 0.0506 / 0.0694 milliseconds. Both components evaluated
361 times for one setup evaluation plus 120 warmup and 240 measured frames.
Those stages account for about 0.66 milliseconds of a 5.89-millisecond scene
frame. They exclude constant-joint preparation, Key/map sinks and downstream
pose, geometry and material work; they are not an exhaustive exclusive profile.

A separate same-binary output-replay diagnostic measured **4.96 milliseconds**
versus **5.28 milliseconds** live native, with identical evaluated outputs.
Replay bypasses control extraction and SDK calculation but still applies the
recorded outputs and performs downstream work. It adds array lookup and an
animated replay index, and was measured once; it is a remaining-work reference,
not a strict theoretical lower bound.

Native-only scaling, one 1,200-frame trial per size:

| Independent characters | Mean milliseconds/frame | Evaluated frames/second |
| ---------------------- | ----------------------: | ----------------------: |
| 1                      |                   5.727 |                  174.61 |
| 2                      |                  10.101 |                   99.00 |
| 4                      |                  16.994 |                   58.84 |
| 8                      |                  31.672 |                   31.57 |

Copies have independent rig, mesh, Key, material and solver data, with identical
actions. Per-character poses matched. These exploratory scaling trials are
not a paired Python scaling study, do not estimate memory limits, and were
recorded before the final formatting/visibility build.

The original profiling script was also run unchanged, with 100 iterations,
20 warmup and imported shape keys. It recorded a 34.813-millisecond
`full_evaluation` mean. Its separately invoked methods overlap and do not
advance frames, so its derived rate must not be compared as viewport FPS or
used as an additive Python-versus-C++ breakdown.

## Implementation

Blender branch: `exp/riglogic-native-perf`, based on
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`.

- `WITH_RIGLOGIC` defaults OFF. The candidate statically links OpenRigLogic
  13.2.7 through its exported CMake package. Both use MSVC 19.44.35217, Ninja,
  Release and the same Windows platform libraries.
- Each graph owns its mutable solver sessions through operation closures.
  No Python per-frame call, SWIG transfer, RNA assignment loop, global solver
  instance or render-to-main-thread queue is used in native mode.
- Constant head transforms are applied before driver evaluation. Completed
  driver bones feed the solve; dynamic output locals are applied before their
  normal Blender bone evaluation. Blender computes final deformation matrices.
- Separate Key and node-tree operations apply evaluated values before geometry
  and material consumers. Explicit component entry/exit nodes preserve existing
  graph relationships. Writes do not recursively tag the graph from workers.
- Experimental persisted ID-property bindings are set once by the setup script.
  `bpy.app.riglogic.rebuild_bindings()` validates and rebuilds them, synchronizing
  evaluated objects and outputs. This is a prototype interface, not a public API.
- Supported switches include global enable, bones, shape keys, texture masks,
  RBF evaluation, component-specific LOD, follow influences and eye visibility.
- Invalid mappings, direct dynamic driver ancestry and shared native output
  ownership are rejected. Arbitrary user-created cross-constraint cycles are
  not fully classified or supported by this validation.
- The reader normalizes DNA coordinates to the add-on's left/up/front frame and
  closes the disk stream after loading. Editor file replacement can therefore
  happen before explicit rebinding without retaining a Windows file lock.

### Intentional Eye-Aim Change

DNA eye-look controls translate the same eye centers from which aim is computed.
The user approved replacing legacy dependency-cycle ordering with a deterministic
bounded solve. It starts from current manual controls each evaluation, uses
current targets and parent transforms, allows at most 12 corrective iterations
after the initial calculation, and checks a `1e-6` control residual.

Ten tested poses converged, matching an iterated Python reference within about
`3.9e-7` in matrices and `2.4e-7` in vertices. Eye constraints outside this
tested setup can fail convergence and are reported, not claimed supported.

### Numerical And Measurement Findings

- Matching matrix order was insufficient: `mathutils` accumulates float products
  in double precision and uses determinant-based inversion. Using other native
  helpers caused RBF-sensitive errors up to `2.9e-5`; matching the reference
  arithmetic fixed them without increasing tolerance.
- The head's `0.0001` master-eye threshold and non-leaf Euler override are part
  of the behavior contract. Omitting either can give visibly different eyes.
- Ordered writes of 82 maps into 41 sockets are intentional in the existing
  mapper. They must not be rejected as competing character writers.
- A first baseline invocation failed to register the Python frame listener and
  looked artificially fast. The output-parity gate rejected it. Native binding
  synchronization initially failed similarly and was also rejected. Neither
  sample is included in reported results.

## Verification

The final scripted suite completed successfully on the final executable:

- 96 SDK parity cases across components, inputs, LODs and reference precision modes.
- All 251 body driven matrices across eight poses; maximum error about `6e-7`.
- All 867 head driven matrices, evaluated mesh vertices and mapped mask values
  across ten manual-eye poses; exact agreement in the recorded test.
- Bounded eye aim and follow-switch cases with convergence assertions.
- Output toggles, RBF-off neutral behavior, visibility, LOD restoration and
  separate evaluated view-layer objects.
- Seven invalid-binding rejection cases and recovery to valid evaluation.
- Independent appended characters, isolated edits, head without body/face board,
  deletion recovery, Windows DNA replacement and explicit reload.
- Undo/redo reconstruction without a manual rebind.
- Native-only Cycles renders at frames 1, 57 and 1, with changed pixels and
  repeated-frame error `1.1920929e-7`; viewport evaluated pose preserved.

Additional verification captured 125 untimed scene snapshots, covering all
120 animation frames, subframes and backward jumps; native poses, all mesh
coordinates and masks matched Python at absolute tolerance `1e-5`. Fresh-process
native-only reload also reproduced six snapshots exactly with the add-on disabled.

Python harnesses pass Ruff 0.14.11. New native C++ files were formatted with
Blender's configuration and compiled in the actual executable. The full unrelated
add-on/Blender test suites were not run.

## Artifacts And Reproduction

Executables are installed separately; the user's installed Blender is unchanged:

```text
E:/repos/build_riglogic_native/candidate-install/blender.exe
E:/repos/build_riglogic_native/baseline-install/blender.exe
```

Final candidate SHA-256:
`6d2a7f5f10cc5aee231bd6f5d8803512f0263cd94bf77638ca1f5e54aa9f53e0`.
The packaged Python SDK remains 13.2.5; its outputs were verified against native
13.2.7. Source and binary provenance must remain attached to any comparison.

- [Final frame summary](../../reports/profiling/native/final_frames_001/summary.json)
- [Final viewport summary](../../reports/profiling/native/final_viewport_001/summary.json)
- [Final verification directory](../../reports/profiling/native/final_verify_003)
- [125-frame native trace result](../../reports/profiling/native/final_verify_002/trace-native.json)
- [Pristine-source baseline](../../reports/profiling/native/completion_001/pristine-python.json)
- [Replay result](../../reports/profiling/native/completion_001/replay-001.json)
- [Scaling result](../../reports/profiling/native/completion_001/scaling.json)
- [Historical profiler results](../../reports/profiling/native/completion_001/legacy)
- [Build logs](../../reports/profiling/native/final_build_002)

Run from the add-on repository in PowerShell; use fresh report directories:

```powershell
& ./scripts/profiling_utils/native_perf.ps1 -Stage BuildCandidate -ReportRoot ./reports/profiling/native/rebuild
& ./scripts/profiling_utils/native_perf.ps1 -Stage Verify -Fixture ./reports/profiling/native/20260905_native_001/ada-lod0-001.blend -NativeFixture ./reports/profiling/native/20260905_native_001/native-reload-fixture.blend -ReportRoot ./reports/profiling/native/reverify
& ./scripts/profiling_utils/native_perf.ps1 -Stage Frames -Fixture ./reports/profiling/native/20260905_native_001/ada-animated-001.blend -ReportRoot ./reports/profiling/native/remeasure -Trials 5 -Warmup 120 -Frames 1200
& ./scripts/profiling_utils/native_perf.ps1 -Stage Viewport -Fixture ./reports/profiling/native/20260905_native_001/ada-animated-001.blend -ReportRoot ./reports/profiling/native/redraw -Trials 3 -Seconds 60 -Shading MATERIAL
```

`BuildSdk` and `Preflight` are also implemented. Frame/viewport reports are
aggregated by [native_frame_report.py](native_frame_report.py). The setup,
diagnostic and test scripts sit next to [native_perf.ps1](native_perf.ps1).
`native_capture_artifacts.py` preserves source patches, new files, build caches
and hashes without staging or committing. The full original plan remains in
session memory; this report describes what was actually implemented and measured.

## Recommended Next Decision

Use **approximately 187 scene-evaluation frames/second and 21 material-preview
draw frames/second** as the measured native reference for this fixture and
machine, not as promises for other scenes.

The next architectural comparison should preserve **in-graph scheduling and
batched evaluated-data writes**. A smaller Blender-native API that lets the
add-on register this evaluator is a plausible way to reduce maintenance while
retaining those gains. A conventional extension called by a Python post-update
handler may remove language overhead but can retain repeated evaluation and
material invalidation costs; benchmark it before assuming equal performance.

Further SDK micro-optimization alone is unlikely to reproduce the 7.35x gain
again. The stage and replay diagnostics suggest examining downstream geometry,
material synchronization and drawing next. These measurements do not identify
one specific shader/GPU operation as the bottleneck, so a GPU rewrite or new
shader architecture is not yet justified by this evidence.

Production integration still needs editor pause/commit/rebind handling,
arbitrary rest/topology edits, broader constraint validation, cancellation and
render-job threading stress, platform coverage, and production-scene profiling.
No commits, pushes, or unrelated asset reverts were made.
