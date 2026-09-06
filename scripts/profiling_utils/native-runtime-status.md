# Native RigLogic Implementation Status

Date: 2026-09-05. Experimental Blender branch: `exp/riglogic-native-perf`.

The agreed Ada-fixture experiment is complete. The authoritative consolidated
report is [Native RigLogic Experiment: Findings](native-runtime-findings.md).
This remains an experimental backend, not a production release.

## Verified

- Built and installed Blender 5.2.1 from base revision `9e2066aef7ef` with
  `WITH_RIGLOGIC=ON`, using MSVC 19.44.35217 and Ninja.
- Executable: `E:/repos/build_riglogic_native/candidate-install/blender.exe`.
- Static OpenRigLogic 13.2.7 is integrated behind a default-OFF CMake option.
- `bpy.app.riglogic` provides diagnostic SDK loading, schema inspection and
  fixed-input evaluation. It is not the real-time entry point.
- SDK output parity: 96 cases passed at absolute tolerance `1e-5`, covering
  head/body, three LODs, eight input poses and packaged default/Float settings.
- Native graph scheduling probe: a constrained input bone drives RigLogic,
  a driven bone, a shape key and an animated material input in one graph pass.
  Eight frame-jump checks pass; original input properties are unchanged.
  Two saved Cycles EXRs prove the map reaches rendered pixels.
- Real Ada body: native constrained-matrix quaternion extraction, SDK solve,
  and rest-relative application to all 251 driven joints are implemented.
  Eight posed-input cases match Python local and evaluated pose matrices
  within `5.96046448e-7`; original native-rig inputs remain unchanged.
- Full head mapping now includes the constant-joint pre-pass, GUI and raw
  inputs, 867 driven joints, five Key targets, and animated texture masks.
  Ten manual-eye poses matched joint matrices, deformed vertices and masks
  exactly in the recorded parity run.
- Eye aim uses the explicitly approved deterministic bounded solve instead
  of legacy cycle-breaking behavior. Ten cases converged within the 13-SDK-call
  limit; maximum tested vertex difference from an iterated reference was
  `2.38e-7`. Follow-switch influences are evaluated before their constraints.
- Output toggles, LOD restoration, independent view layers, invalid-binding
  rejection/recovery, two-character isolation, head-only/no-face-board operation,
  character deletion, and Windows DNA replacement/reload have passed targeted tests.
- Persisted native bindings reopen and reproduce six frame snapshots exactly
  in a fresh Blender with the Character DNA add-on disabled.
- Prepared an Ada LOD0 fixture with head/body/face board, five shape-key
  data blocks, and the texture-logic node. No source DNA was modified.
- Preflight, pinned Python lint, native compilation and the final focused
  verification suite pass, including undo/redo, repeated renders and visibility.
- A 125-frame/subframe animation trace passes all-output parity.

## Measured Performance

These are experimental, workload-specific results, not a universal maximum.
The fixture has Ada LOD0 geometry, face-board animation and body-driver-bone
animation, shape keys and animated maps. It does not contain a separate body
control rig or production skin albedo/normal textures; it uses the imported
topology images and mask atlas. Preserve that distinction when quoting results.

| Measurement                          |         Python |          Native | Scope                                 |
| ------------------------------------ | -------------: | --------------: | ------------------------------------- |
| Mean scene frame time (milliseconds) |          39.39 |            5.36 | Five paired trials, 1,200 frames each |
| Scene evaluation rate                | 25.39 frames/s | 186.63 frames/s | No viewport drawing                   |
| Scene p95 frame time (milliseconds)  |          42.68 |            6.85 | Raw frame samples                     |
| Material-preview draw rate           | 12.19 frames/s |  20.97 frames/s | Three 60-second captures each         |

The recorded paired scene speedup is **7.35x**, with final-frame output parity
within `5.96e-7`. The viewport observer records post-pixel draw completion,
not monitor presentation; presented FPS remains unmeasured. Window/region
dimensions and exact binary/fixture hashes are included in each report.

A later single-trial diagnostic measured 5.28 milliseconds live native versus 4.96 milliseconds
with prerecorded same-frame outputs. Replay preserves joint/shape/material
application and downstream scene work, and matched the output snapshot exactly.
It has its own lookup/index overhead and is not a mathematical upper bound.

Native-only scaling, one 1,200-frame trial per size: 1 character 174.61,
2 characters 99.00, 4 characters 58.84, and 8 characters 31.57 evaluated
frames/s. Copies have independent data and solver state but identical actions.

The SDK and candidate both use MSVC 19.44.35217. A separately built pristine
source baseline measured 40.26 milliseconds per Python frame and passed output
parity against the native candidate.

## Evidence

Artifacts are under `reports/profiling/native/20260905_native_001/`:

- `sdk-parity-001.json`: 96-case SDK comparison.
- `graph-probe-001.json`: one-pass scheduling checks and rendered pixel values.
- `graph-probe-001-frame-1.exr`, `graph-probe-001-frame-4.exr`: rendered outputs.
- `ada-lod0-001.blend`, `ada-lod0-001.json`: fixture and dependency audit.
- `body-parity-003.json`: finite-value-checked real-body parity.
- `body-parity-001.json`: retained failing result before correcting arithmetic.
- `head-parity-004.json`: full manual-eye head geometry and mask parity.
- `head-eye-parity-001.json`: approved bounded eye-aim convergence and parity.
- `native-reload-result.json`: fresh-process native-only persistence.
- `settings-001.json`, `bindings-001.json`, `isolation-001.json`: settings and lifecycle gates.

Additional result directories:

- `reports/profiling/native/final_frames_001/summary.json`: final five paired scene trials.
- `reports/profiling/native/final_viewport_001/`: final three paired viewport trials and screenshots.
- `reports/profiling/native/final_verify_003/`: passing final focused verification suite.
- `reports/profiling/native/final_verify_002/trace-native.json`: 125-frame/subframe parity.
- `reports/profiling/native/completion_001/`: output replay, scaling and the unchanged legacy profiler run.

Updated successful preflight:
`reports/profiling/native/20260905_native_002/preflight.json`.

## Reproduction

Run the following in PowerShell, one invocation at a time. Reusing an output
file is rejected; choose a new name for each verification run.

```powershell
& E:/repos/build_riglogic_native/candidate-install/blender.exe --background --factory-startup --python-exit-code 1 --python E:/repos/character-dna-addon/scripts/profiling_utils/native_benchmark.py -- verify-sdk --output E:/repos/character-dna-addon/reports/profiling/native/sdk-parity-new.json
```

```powershell
& E:/repos/build_riglogic_native/candidate-install/blender.exe --background --factory-startup --python-exit-code 1 --python E:/repos/blender/tests/python/bl_riglogic.py -- --dna E:/repos/character-dna-addon/tests/test_files/dna/ada/head.dna --report E:/repos/character-dna-addon/reports/profiling/native/graph-probe-new.json
```

```powershell
& E:/repos/build_riglogic_native/candidate-install/blender.exe --background --factory-startup --python-exit-code 1 --python E:/repos/character-dna-addon/scripts/profiling_utils/native_body_test.py -- --fixture E:/repos/character-dna-addon/reports/profiling/native/20260905_native_001/ada-lod0-001.blend --output E:/repos/character-dna-addon/reports/profiling/native/body-parity-new.json
```

## Findings That Affect the Port

- The `mathutils` matrix product accumulates float products in double
  precision. Its inverse divides `adjoint_m4_m4` results by the determinant. Using the
  usual native matrix helpers produced body RBF differences up to `2.9e-5`;
  matching the reference arithmetic fixed the tested cases without loosening
  tolerance. Singular-matrix fallback equivalence still needs coverage.
- Adding operations to Key geometry or node-tree output components requires
  explicit entry/exit nodes; the former single-operation inference no longer
  applies. The prototype preserves normal geometry and material consumers.
- Newly added shape keys start at value 1.0 in this Blender version. Tests
  must explicitly establish their initial value.
- Head drivers descend from `spine_04` and `spine_05`, which the Python mapper
  writes. Those spine joints have no variable joint-group output rows in Ada.
  A constant pre-pass is needed before head-driver evaluation; a blanket
  driver-done -> solve -> all-output-bones ordering would create a cycle.
- Blender's GitHub mirror did not provide the required source LFS objects.
  A command-scoped canonical LFS endpoint fetched the build assets without
  changing remotes: `https://projects.blender.org/blender/blender.git/info/lfs`.
- Developer-shell INCLUDE/LIB settings must be restored after a terminal is
  cleaned up. CMake's Visual Studio generator selected the older installed
  toolset; Ninja in the initialized 14.44 developer shell selected correctly.

## Qualification Limits

Production qualification remains outside the demonstrated fixture: arbitrary
cross-constraint cycles, constrained eye bones, interactive render cancellation,
editor integration and complete invalidation on arbitrary topology/rest edits
need more coverage before this is a supported backend.

The user approved finishing with the verified Ada fixture. A production-textured
control-rig scene and presentation capture would be needed to claim production
viewport or monitor-presented FPS. Existing results must not be relabeled as either.

The internal ID-property binding schema is an experimental test interface.
Use `bpy.app.riglogic.rebuild_bindings()` after editing it. Validation failures
raise an explicit setup error and prevent partial native graph registration.
Build and verification stages are available in `native_perf.ps1`; use a fresh
report directory for every run. Source files and generated reports remain
uncommitted for inspection.

No commits or pushes have been made, and installed Blender was not replaced.
