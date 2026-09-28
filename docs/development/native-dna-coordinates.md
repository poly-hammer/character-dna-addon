# Native coordinates and DNA-only hand alignment

This change addresses [issue #368](https://github.com/poly-hammer/character-dna-addon/issues/368)
with a Blender-side conversion correction. Conversion fits the custom anatomy,
then bakes the hands into the base DNA's reference orientation. The exported
DNA works with MetaHuman Creator's default retargeter; no Unreal script or
custom retargeter is part of the workflow.

## Using the fix

1. Install the matching add-on and rebuilt OpenRigLogic Blender module, then
   restart Blender. Both Windows Python 3.11 and 3.13 builds were verified.
   A Python-only reload cannot replace an already loaded native module.
2. For existing scenes, use **Migrate Legacy Data → Migrate Now**. The operator
   re-reads each rig's DNA, replaces its rest frames in place, and rebuilds its
   runtime drivers. Meshes, materials and character object references remain.
   Changing a root rotation or coordinate-version tag is not a migration.
   To redo mesh conversion, convert the wrapped meshes again instead.
3. In MetaHuman Creator, use **From DNA > Replace** with the converted DNA.
4. Play the animation using the normal MetaHuman preview.

The converted neutral hand pose changes: the wrist orientation and finger
directions follow the base DNA. Fitted finger lengths, surface detail, and
corrective offsets are carried through using the existing skin weights. Wrist
positions and body joints outside the hand hierarchy are preserved. Vertices
without hand-bone influence are unchanged by this alignment, including the
custom waist and hip pose. The artist's input mesh objects are not modified.

Use a base DNA whose hand reference pose is compatible with the animation
source. This does not replace weight/corrective sculpting for arbitrary custom
anatomy. The separate elbow problem mentioned in the report was not reproduced.

## Upgrading existing animation

Migration lists Actions assigned to the DNA armatures and attached body control
rig, including NLA clips, and asks whether to migrate them. Original Actions
are retained; upgraded copies are assigned only to the upgraded character.
Face-board animation keeps its existing Action because its UI axes have not
changed. Declining animation migration detaches the active Action and mutes
existing NLA clips on the upgraded rigs, retaining them for recovery.

Quaternion animation under the usual basis exchange preserves its original
keys, handles and interpolation. Euler animation and control-rig rebakes are
sampled every frame over each Action's range, including fractional authored
keyframes. NLA clip timing, repeats and blend settings are preserved.

An attached control rig requires removal before migration unless **Character
Assembly** and **Character Control Rig** are enabled and its builder settings
are available. In that case the dialog offers an explicit rebuild confirmation:
it saves a uniquely named `.pre-migration-….blend` backup beside the current
file (or in Blender's temporary directory for an unsaved scene), stores the
original Actions, rebuilds the controls and rebakes animation copies. The dialog
warns that custom control-rig edits may be lost. Evaluated body-driver motion is
checked before replacing the old controls; a failed check restores the original
rigs and animation. The backup path is reported by the operator.

Control-rig motion is checked separately for bone-endpoint displacement
(0.1 mm), rotation (0.1 degrees), and scale (0.01%). Displacement accounts for
the armature transform and scene units. Re-reading DNA can slightly alter short
finger axes without meaningfully moving the surface; comparing individual matrix
coefficients incorrectly rejected such rigs. Errors now identify the bone,
frame, and measured differences. These checks compare control-driven motion;
restoring previously inactive RigLogic correctives can separately change the
deformed mesh.

Leave NLA Tweak Mode before migrating. Animated influence/time warps and meta
strips on control rigs must first be baked to Actions, as must custom drivers or
constraints on DNA bones. These cases are rejected before replacing the rigs.

The implementation lives in `utilities/migration.py`: legacy-data discovery,
preflight checks, rig and Action upgrades, control-rig rebuilding, and rollback
share one module. The operator calls this module directly; existing
package-level migration utilities remain available to integrations.

## Why bone axes alone are insufficient

Unreal's FK-chain retargeting applies a source rotation relative to the source
retarget pose onto the target retarget pose:

```text
rotation_delta = source_current * inverse(source_initial)
target_current = rotation_delta * target_initial
```

See `Engine/Plugins/Animation/IKRig/Source/IKRig/Private/Retargeter/RetargetOps/FKChainsOp.cpp`,
including its source-parent compensation. The stock MetaHuman retargeter's
finger chains use `Interpolated` rotation and `None` translation. Its global
rotation delta is applied even when a wrapped hand points in a different
anatomical direction. With the target's neutral pose as its retarget pose,
the neutral rotation cancels against the inverse bind during skinning:

```text
skin_rotation = rotation_delta * neutral_rotation * inverse(neutral_rotation)
              = rotation_delta
```

The reported curl is already visible in the raw retargeted pose before the
RigLogic post-process. Correcting RBF evaluation or changing the DNA coordinate
basis does not supply the missing hand alignment. The earlier Unreal alignment
script was an A/B diagnostic; it has been removed from the supported workflow.

The converter now fits anatomical hand/finger frames, reconstructs their bind
transforms in the template orientation, and skins the converted neutral hand
vertices by `aligned_bind * inverse(fitted_bind)`. Both sides of the skinning
contract are therefore written to DNA together. Calibration updates normals
and propagates the result to lower LODs. This is a physical neutral-hand pose
change, separate from the passive coordinate-system migration below.

## Coordinate contract

| Boundary | Axes / rotation sequence | Units |
| --- | --- | --- |
| In-memory DNA | left, back, up / XZY | centimetres, degrees |
| Blender scene | native Z-up joint and mesh basis | metres, radians |
| Exported binary or JSON DNA | left, up, front / XYZ | centimetres, degrees |

`dna_io/coordinates.py` supplies `CoordinateSystemTransformPolicy_Transform`
to the SDK reader. The Maya-to-Blender mapping is `(x, y, z) -> (x, -z, y)`.
Joint rotations undergo the true change of basis `C R C^-1`, including local
driver quaternions, RBF data, and the other supported DNA layers. The previous
root/mesh 90-degree bridge is removed. Unit conversion remains separate.

XZY is intentional: it is Maya's XYZ rotation sequence expressed in the new
basis. Keeping XYZ after exchanging the Y/Z axes changes the meaning of
additive facial rotation channels. Pose-bone RNA may use XYZ or quaternion
storage; the adapter converts the represented rotation at that boundary.

Binary and JSON writers stage edits in memory, use the SDK to convert back
to canonical Y-up/XYZ, and then write the destination. JSON readers also pass
through a configured binary reader, since the JSON API has no configuration
overload. Unknown layers are preserved by the SDK's preserve policy.

Imported armatures carry `dna_coordinate_version = 1`; native bindings use
schema 3 and require the `blender_coordinates` capability. Legacy rigs fail
preflight with an actionable reimport message rather than evaluating mixed axes.
Bundled Maya-local pose presets and imported FBX animation are converted at
their boundaries. Historical snapshot files remain unchanged.

## RBF and converter changes

- Normalize blended SDK output quaternions before making Blender matrices.
  Non-unit output quaternions otherwise introduce unwanted scale and rotation.
- Preview all RBF solver contributions together, including the half-finger
  companions, while writing only the selected pose's edited contribution.
  Repeated Apply operations preserve untouched coefficients.
- Preserve joint membership when coordinate conversion prunes zero rows, and
  insert newly authored sparse channels beside that joint's existing rows.
  The SDK coordinate converter requires contiguous joint blocks.
- Fit finger frames using skin-weighted template/wrapped vertex correspondences
  for roll and fitted child origins for segment direction. Corrective frames
  follow their owning segment; degenerate samples use the neighboring segment
  or hand rotation. The converter then bakes hand reference alignment into both
  the neutral mesh and skeleton. Standalone bone fitting still retains fitted
  origins; the normalization runs only during full body conversion.

The modern SDK's finger twist-distance behavior agrees with the Unreal
snapshots. The RBF distance algorithm was not changed to match the older Maya
finger captures. Maya tests retain their finger exclusion; the separate Unreal
suite covers every captured finger pose in both runtime and editor preview.

## Development and validation

The implementation spans the main add-on, the Pro `editors` submodule, and
`OpenRigLogic/modules/blender`. Rebuild/package the native module for every
supported platform and Python version with the same source changes. The local
Windows builds are `build/blender45` (Python 3.11) and `build/blender52`
(Python 3.13); other platform binaries require their normal release builders.

Relevant regression suites include `test_body_rig_logic_unreal.py`,
`test_body_rbf_editor.py`, `test_rbf_edits.py`, `test_dna_coordinates.py`,
`converter/test_finger_frames.py`, `test_fbx_writer.py`, and
`test_native_runtime.py`, plus the head export, calibration, and raw-control
editor checks. Native CTest covers quaternion normalization and the XZY adapter.
The committed Maya and Unreal snapshot data is preserved; only the test input
boundary converts historical Maya-local driver values to the new basis.

Earlier coordinate/RBF verification (Windows, September 25, 2026):

| Check | Result |
| --- | --- |
| RBF edits, Unreal runtime/editor poses, Maya editor round trips, native lifecycle | 622 passed, 1 skipped |
| Installed-loader converter, sparse finger edit, binary/JSON coordinates, FBX | 35 passed |
| Head export/calibration and sampled raw-control editor regressions | 298 passed, 9 skipped |
| Maya non-finger runtime/editor and facial snapshot regression run | 273 passed |
| Native CTest, Python 3.11 and 3.13 | 8 tests passed per build |
| Stock Blender 4.5 installed-module neutral and finger smoke check | 3 poses passed; maximum joint-origin error 0.00000968 m |

These runs overlap and are not a count of unique tests. The head editor run
used the repository's CI sampling setting. Changed production source and new
test files pass Ruff; changed lines introduce no Pyright errors, although
existing repository type diagnostics remain. The documentation builds with
MkDocs. Unreal animation checks are engine integration checks, separate from
the snapshot unit tests.

DNA-only converter verification (Windows, September 28, 2026):

- Converter and coordinate regression suites: **34 passed**, including hand
  alignment, preserved non-hand vertices, and end-to-end DNA conversion.
- The actual converter processed the issue reproduction scene. Its DNA was
  imported with whole-rig Replace in Unreal 5.8.3, using the unchanged
  `/MetaHumanCharacter/Animation/Retargeting/RTG_MH_IKRig` asset.
- All 600 AngerB frames were evaluated with `ABP_Body_PostProcess`: 342 bones
  per frame, all transforms finite. Both hands were rendered at frames 0, 159,
  252, 290, and 599; the reproduced finger curl is corrected.
- Exporting that DNA back out of Unreal produced **zero differences** in mesh
  positions, neutral joint translations, and neutral joint rotations.
- Across all four body LODs, vertices outside hand skin influence differed
  from the original conversion by at most **0.0000113 cm**, numerical export
  precision. The neutral hands are intentionally reposed; their maximum
  vertex displacement in this reproduction is **11.52 cm**.
- Ruff passes for changed converter code and tests; Pyright reports no errors
  in new/changed lines. MkDocs builds successfully. The committed Maya and
  Unreal snapshot data remains untouched.
