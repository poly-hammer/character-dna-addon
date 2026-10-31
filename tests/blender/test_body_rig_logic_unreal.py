"""Body runtime parity with poses baked by Unreal, including every finger solver."""

import hashlib
import json
import math

import bpy
import pytest

from mathutils import Matrix, Quaternion

from constants import TEST_DNA_FOLDER, TEST_FBX_FOLDER, TEST_FILES_FOLDER, TEST_JSON_POSES_FOLDER
from utilities.bones import get_bone_differences


SNAPSHOTS = TEST_FILES_FOLDER / "json" / "poses_unreal" / "ada_body_rig"
DNA = TEST_DNA_FOLDER / "ada_unreal" / "body.dna"
INPUTS = json.loads((SNAPSHOTS / "inputs.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def unreal_body(rig_factory, tmp_path_factory, import_lods, import_shape_keys):
    """Own the Unreal capture rig; never borrow or restore another module's scene."""
    from fixtures.rigs import RigProfile

    profile = RigProfile(
        components=("body",),
        asset="ada_unreal",
        materials=True,
        lods=tuple(int(lod[-1]) for lod in import_lods),
        shape_keys=import_shape_keys,
    )
    with rig_factory.lease(profile, tmp_path_factory.mktemp("unreal_body")) as instance:
        assert instance.body_rig
        instance.auto_evaluate_body = False
        yield instance


def test_unreal_snapshot_provenance():
    """Reject stale inputs, missing poses, or an FBX from a different capture."""
    provenance = json.loads((SNAPSHOTS / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["riglogic_version"] == "13.2.9"
    assert set(provenance["riglogic_source_sha256"]) == {"RBFSolver.cpp", "InterpolativeRBFSolver.cpp"}
    assert hashlib.sha256(DNA.read_bytes()).hexdigest() == INPUTS["dna_sha256"] == provenance["dna_sha256"]
    manifest = (SNAPSHOTS / "inputs.json").read_text(encoding="utf-8").encode("utf-8")
    assert hashlib.sha256(manifest).hexdigest() == provenance["inputs_sha256"]
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


def test_capture_inputs_use_canonical_coordinates(tmp_path):
    """The generator must reproduce Maya inputs even though the addon reads Z-up DNA."""
    from snapshot_tools.unreal_body.prepare_inputs import prepare

    path = tmp_path / "inputs.json"
    prepare(DNA, path)
    generated = json.loads(path.read_text(encoding="utf-8"))
    assert generated.keys() == INPUTS.keys()
    assert {key: value for key, value in generated.items() if key != "joints"} == {
        key: value for key, value in INPUTS.items() if key != "joints"
    }
    for actual, expected in zip(generated["joints"], INPUTS["joints"], strict=True):
        assert actual["name"] == expected["name"]
        assert actual["parent"] == expected["parent"]
        assert actual["unreal_translation_cm"] == expected["unreal_translation_cm"]
        # mathutils' float32 trigonometry can differ by an ULP across platforms.
        assert actual["unreal_rotation_xyzw"] == pytest.approx(expected["unreal_rotation_xyzw"], abs=1e-6)


@pytest.mark.parametrize("angle", [30, 60, 90])
def test_finger_half_solver_uses_unwrapped_twist_distance(unreal_body, angle):
    """Positive twists interpolate toward +120, without wrapping to the -120 target."""
    from character_dna.bindings import riglogic
    from character_dna.dna_io import release_dna_handle

    reader = unreal_body.body_dna_reader
    manager = unreal_body.body_manager
    state = riglogic.RigInstance(rigLogic=manager, memRes=None)
    try:
        rotation = Quaternion((0, -1, 0), math.radians(angle))
        for index in range(reader.getRawControlCount()):
            name, axis = reader.getRawControlName(index).rsplit(".", 1)
            value = rotation["wxyz".index(axis[-1])] if name == "index_01_l" else float(axis == "qw")
            state.setRawControl(index, value)
        manager.calculate(state)
        solver = next(
            index
            for index in range(reader.getRBFSolverCount())
            if reader.getRBFSolverName(index) == "index_l_01_half_UERBFSolver"
        )
        control_offset = state.getRawControlCount() + state.getPSDControlCount()
        weights = {
            reader.getRBFPoseName(pose): [
                state.getRBFControl(index - control_offset) for index in reader.getRBFPoseOutputControlIndices(pose)
            ]
            for pose in reader.getRBFSolverPoseIndices(solver)
        }
        assert weights["index_l_01_half_curl_120"] == pytest.approx([angle / 120], abs=2e-5)
        assert weights["index_l_01_half_push_120"] == pytest.approx([0], abs=2e-5)
        assert weights["index_l_01_half_caps_020"] == pytest.approx([0], abs=2e-5)
    finally:
        release_dna_handle(state)


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
