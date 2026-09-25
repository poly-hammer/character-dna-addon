# Unreal body reference snapshots

The Maya snapshots in `tests/test_files/json/poses/ada_body_rig` and their original DNA are preserved. Their tests exclude fingers because those captures use the older Maya/PoseDriver behavior. `test_body_rig_logic_unreal.py` evaluates every Unreal frame, including all fingers, without CI subsampling or snapshot rewriting during tests.

The new references contain 215 frames at 30 fps: neutral plus 214 solver targets (92 non-finger, 122 finger), each containing all 342 joint origins in Blender world axes, in meters. Expected outputs come from the assembled Ada Blueprint's body post-process at LOD0, baked into an AnimSequence, exported as FBX, and extracted with UFBX. Blender never generates expected outputs.

## Matching character data

`tests/test_files/dna/ada_unreal/body.dna` was exported using **Export DNA** on `/Game/MetaHumans/Ada/Body/SKM_Ada_BodyMesh_DNA` in Unreal 5.8.3. The original Ada fixture was exported with Unreal 5.6. Its neutral joints differ from this assembled mesh by up to 0.226112 cm, so using that older DNA for the new references would create unrelated failures. The legacy fixture remains unchanged.

`inputs.json` records the exact DNA hash, frame-to-solver mapping, neutral transforms, and driver quaternions. Targets come from the DNA's stored quaternion controls, not an interpretation of pose-name Euler suffixes. Driver deltas retain their Maya-local meaning and are composed as `neutral * delta` after reflection into Unreal coordinates. This does not change the addon's coordinate handling.

`provenance.json` records Unreal version, source asset paths, capture hashes, and audit errors. The original baked FBX is retained at `tests/test_files/fbx/ada_unreal_rbf.fbx` under Git LFS. All translation, rotation, and scale tracks remain in that FBX; the JSON assertions use joint origins at the existing 1 mm tolerance.

## Regeneration

1. Export the DNA from the assembled **body DNA asset** using its Content Browser **Export DNA** action. Place that exact file at `tests/test_files/dna/ada_unreal/body.dna`. Keep the legacy Maya DNA intact. Do not invoke Creator's `export_dna` on an unopened/uninitialized parametric character: this engine build exited during that operation.
2. In the repository's bpy-enabled Python environment:

   ```powershell
   .venv\Scripts\python.exe tests/snapshot_tools/unreal_body/prepare_inputs.py tests/test_files/dna/ada_unreal/body.dna tests/test_files/json/poses_unreal/ada_body_rig/inputs.json
   ```

3. In the open Unreal Editor's Python interpreter (or the project's Python RPC), run the following, substituting the checkout path:

   ```python
   import sys
   sys.path.insert(0, r"E:\repos\character-dna-addon\tests\snapshot_tools\unreal_body")
   import bake_unreal
   bake_unreal.bake(
       r"E:\repos\character-dna-addon\tests\test_files\json\poses_unreal\ada_body_rig\inputs.json",
       r"E:\repos\character-dna-addon\scratches\unreal-body-capture",
   )
   ```

   This updates assets only under `/Game/CharacterDNATests/AdaRBF`: the driver clip, baked clip, compression settings, and review Level Sequence. It creates/reuses its tagged Ada preview actor in the current level. It does not save the user's level. The AnimSequences are self-contained; rerun the capture to rebind the review Level Sequence after reopening an unsaved level.

   The input clip contains one extra terminal guard key to prevent Sequencer wrapping the final pose to neutral. Linear interpolation avoids stepped-key frame-boundary errors. Both clips use the actual Ada mesh as retarget source and bitwise compression, avoiding shared-skeleton proportion offsets and ACL key approximation. The bake includes the actual Blueprint post-process; viewing the already-baked clip should disable post-process evaluation to avoid applying correctives twice.

4. Extract and validate with UFBX:

   ```powershell
   .venv\Scripts\python.exe tests/snapshot_tools/unreal_body/extract_fbx.py tests/test_files/json/poses_unreal/ada_body_rig/inputs.json scratches/unreal-body-capture tests/test_files/json/poses_unreal/ada_body_rig tests/test_files/fbx/ada_unreal_rbf.fbx
   ```

   The extractor samples exact `frame / fps` times, includes animated scale, and implements Unreal FBX's component-wise scale inheritance (RrSs). It checks every resulting joint against a separate raw-pose audit read from Unreal's baked animation. A mismatch above 0.01 mm aborts before writing snapshots. The large intermediate audit stays in the scratch capture directory.

5. Run both baselines:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_body_rig_logic.py tests/test_body_rig_logic_unreal.py -o addopts="" -q
   ```

The new suite sets only driver bone quaternions and evaluates the runtime. It deliberately avoids the RBF editor's isolated-pose preview, which is a different test subject. The existing Maya runtime and editor tests remain in place.

## Capture validation (2026-09-25)

- Combined Maya and Unreal suites: **400 passed** (184 Maya cases, 215 Unreal poses, one provenance check).
- All 92 non-finger and 122 finger poses pass without changing production code or the 1 mm tolerance.
- Maximum UFBX extraction difference from Unreal's raw pose audit: **0.00000114754 m** (0.00115 mm).
- Maximum baked driver rotation difference from the authored input: **0.003495 degrees**.
- A second complete Unreal bake and UFBX extraction reproduced all 215 JSON snapshots exactly. FBX hashes differ between exports because the file contains export metadata.
- The original Maya JSON snapshots and `dna/ada` fixture have no Git diff.

UFBX's inheritance definitions are documented in its [public header](https://github.com/ufbx/ufbx/blob/master/ufbx.h). The bundled Python wrapper omits enum value 2; the extractor handles that specific omission without changing the installed library.
