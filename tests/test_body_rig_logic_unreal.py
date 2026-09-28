"""Body runtime parity with poses baked by Unreal, including every finger solver."""

import hashlib
import json

import bpy
import pytest

from mathutils import Matrix, Quaternion

from character_dna.ui.callbacks import get_active_rig_instance
from constants import BODY_DNA_FILE, TEST_DNA_FOLDER, TEST_FBX_FOLDER, TEST_FILES_FOLDER, TEST_JSON_POSES_FOLDER
from fixtures.scene import load_dna
from utilities.bones import get_bone_differences


SNAPSHOTS = TEST_FILES_FOLDER / "json" / "poses_unreal" / "ada_body_rig"
DNA = TEST_DNA_FOLDER / "ada_unreal" / "body.dna"
INPUTS = json.loads((SNAPSHOTS / "inputs.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def unreal_body(load_body_dna, import_lods, import_shape_keys):
    """Use the DNA exported from the exact assembled mesh; restore the Maya fixture."""
    settings = {
        "import_lods": import_lods,
        "import_shape_keys": import_shape_keys,
        "import_face_board": False,
        "include_body": False,
    }
    load_dna(file_path=DNA, **settings)
    instance = get_active_rig_instance()
    assert instance and instance.body_rig
    instance.auto_evaluate_body = False
    try:
        yield instance
    finally:
        # load_body_dna is session-scoped: subsequent Maya tests must get its rig back.
        load_dna(file_path=BODY_DNA_FILE, **settings)


def test_unreal_snapshot_provenance():
    """Reject stale inputs, missing poses, or an FBX from a different capture."""
    provenance = json.loads((SNAPSHOTS / "provenance.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(DNA.read_bytes()).hexdigest() == INPUTS["dna_sha256"] == provenance["dna_sha256"]
    assert hashlib.sha256((SNAPSHOTS / "inputs.json").read_bytes()).hexdigest() == provenance["inputs_sha256"]
    assert (
        hashlib.sha256((TEST_FBX_FOLDER / provenance["fbx_file"]).read_bytes()).hexdigest() == provenance["fbx_sha256"]
    )
    assert provenance["maximum_fbx_audit_error_m"] < 1e-5
    assert [frame["frame"] for frame in INPUTS["frames"]] == list(range(len(INPUTS["frames"])))
    actual = {(frame["solver"], frame["pose"]) for frame in INPUTS["frames"] if frame["solver"]}
    legacy = {(path.parent.name, path.stem) for path in (TEST_JSON_POSES_FOLDER / "ada_body_rig").glob("*/*.json")}
    assert actual == legacy  # Explicitly retains all previously excluded finger targets.
    assert len(actual) + 1 == provenance["frames"] == len(INPUTS["frames"])
    assert len(INPUTS["joints"]) == provenance["joints_per_frame"]


@pytest.mark.parametrize("editing", [False, True], ids=["runtime", "editor"])
@pytest.mark.parametrize("frame", INPUTS["frames"], ids=[frame["pose"] for frame in INPUTS["frames"]])
def test_body_pose_unreal(unreal_body, frame, editing):
    """Replay only the authored drivers, then compare all evaluated joint origins."""
    instance = unreal_body
    rig = instance.body_rig
    for bone in rig.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    for name, (x, y, z, w) in frame["drivers"].items():
        bone = rig.pose.bones[name]
        bone.rotation_mode = "QUATERNION"
        # The capture manifest records Maya-local quaternions. Express them in
        # the Blender-native joint basis without changing the reference data.
        bone.rotation_quaternion = Quaternion((w, x, -z, y))
    bpy.context.view_layer.update()
    if editing and frame["solver"]:
        from character_dna.editors.rbf_editor.utilities import update_pose
        from character_dna.utilities import collection_to_list
        from utilities.rbf_editor import set_body_pose

        set_body_pose(frame["solver"], frame["pose"])
        before = collection_to_list(instance.rbf_editor.rbf_solver_list)
        # Applying an unchanged contextual preview must not bake its companions
        # into this pose, including after a repeated Apply.
        update_pose(instance, bpy.context)
        update_pose(instance, bpy.context)
        assert collection_to_list(instance.rbf_editor.rbf_solver_list) == before
    else:
        instance.evaluate(component="body")
    path = SNAPSHOTS / (frame["solver"] or "_neutral") / (frame["pose"] + ".json")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert set(expected) == {joint["name"] for joint in INPUTS["joints"]}
    differences, _ = get_bone_differences(rig.name, target_bone_locations=expected, tolerance=0.001)
    assert not differences, f"{frame['pose']}: {sorted(differences, key=lambda difference: -difference[1])}"
