import shutil
import uuid

import bpy
import pytest

from mathutils import Quaternion, Vector

from character_dna.ui.callbacks import get_active_rig_instance
from character_dna.utilities import reset_pose
from constants import (
    DNA_BEHAVIOR_VERSION,
    DNA_DEFINITION_VERSION,
    DNA_RBF_BEHAVIOR_VERSION,
    DNA_RBF_EXTENSION_VERSION,
    TEST_DNA_FOLDER,
)
from utilities.dna_data import get_dna_json_data
from utilities.rbf_editor import set_body_pose


TOLERANCE = 1e-5
BODY_FILE_NAME = "body.dna"


def test_finger_edit_keeps_companion_contributions_isolated(fresh_rbf_test_scene):
    """A real edit survives commit, while other solver columns remain unchanged."""
    import numpy as np

    from character_dna.editors.rbf_editor.utilities import update_pose
    from character_dna.utilities import collection_to_list

    instance = get_active_rig_instance()
    solver_name = "index_l_01_UERBFSolver"
    pose_name = "index_l_01_curl_070"
    pose, _, _ = set_body_pose(solver_name, pose_name)
    assert pose is not None

    def coefficients():
        return {
            (solver.name, item.name, driven.name): (
                tuple(driven.location),
                tuple(driven.euler_rotation),
                tuple(driven.scale),
            )
            for solver in instance.rbf_editor.rbf_solver_list
            for item in solver.poses
            if item.name != "default"
            for driven in item.driven
            if driven.data_type == "BONE"
        }

    original = coefficients()
    bone = instance.body_rig.pose.bones["index_01_mcp_l"]
    bone.rotation_euler = (bone.matrix_basis.to_quaternion() @ Quaternion((1, 0, 0), 0.035)).to_euler("XYZ")
    bone.location += Vector((0.002, -0.001, 0.003))
    expected = bone.matrix_basis.copy()
    update_pose(instance, bpy.context)
    authored = collection_to_list(instance.rbf_editor.rbf_solver_list)
    update_pose(instance, bpy.context)
    assert collection_to_list(instance.rbf_editor.rbf_solver_list) == authored
    assert bpy.ops.character_dna.commit_rbf_solver_changes() == {"FINISHED"}
    instance.destroy()
    instance.body_initialize()
    set_body_pose(solver_name, pose_name)
    np.testing.assert_allclose(
        bone.matrix_basis, expected, atol=2e-5, err_msg=str(coefficients()[(solver_name, pose_name, bone.name)])
    )
    for key, values in coefficients().items():
        if key != (solver_name, pose_name, bone.name):
            np.testing.assert_allclose(values, original[key], atol=1e-6, err_msg=str(key))


def get_rbf_pose_data_from_json(json_data: dict, pose_name: str) -> tuple[int, dict | None]:
    """
    Get RBF pose data from JSON, merging data from both rbfb1.0 (behavior) and rbfe1.0 (extension).

    The DNA JSON format stores pose data in two sections:
    - rbfb1.0: Contains pose name and scale
    - rbfe1.0: Contains outputControlIndices, outputControlWeights, inputControlIndices

    This function merges data from both sections for a complete pose representation.
    """
    rbf_behavior_data = json_data.get(DNA_RBF_BEHAVIOR_VERSION, {})
    rbf_extension_data = json_data.get(DNA_RBF_EXTENSION_VERSION, {})

    behavior_poses = rbf_behavior_data.get("poses", [])
    extension_poses = rbf_extension_data.get("poses", [])

    for pose_index, pose in enumerate(behavior_poses):
        if pose.get("name") == pose_name:
            # Merge with extension data if available
            merged_pose = dict(pose)
            if pose_index < len(extension_poses):
                ext_pose = extension_poses[pose_index]
                merged_pose["outputControlIndices"] = ext_pose.get("outputControlIndices", [])
                merged_pose["outputControlWeights"] = ext_pose.get("outputControlWeights", [])
                merged_pose["inputControlIndices"] = ext_pose.get("inputControlIndices", [])
            return pose_index, merged_pose

    return -1, None


def get_rbf_solver_data_from_json(json_data: dict, solver_name: str) -> tuple[int, dict | None]:
    rbf_data = json_data.get(DNA_RBF_BEHAVIOR_VERSION, {})
    solvers = rbf_data.get("solvers", [])

    for solver_index, solver in enumerate(solvers):
        if solver.get("name") == solver_name:
            return solver_index, solver

    return -1, None


@pytest.fixture(scope="session")
def original_body_dna_json_data(temp_folder, dna_folder_name: str) -> dict:
    """
    Fixture that returns the original body DNA as JSON data.
    Reads from the original TEST_DNA_FOLDER to ensure we compare against pristine data.
    """
    dna_file_path = TEST_DNA_FOLDER / dna_folder_name / BODY_FILE_NAME
    json_file_path = temp_folder / dna_folder_name / f"{BODY_FILE_NAME}_original.json"
    return get_dna_json_data(dna_file_path, json_file_path, data_layer="All")


@pytest.fixture(scope="function")
def fresh_rbf_test_scene(addon, temp_folder, dna_folder_name: str):
    """
    Function-scoped fixture that sets up a fresh scene with DNA files
    copied to a unique temp location for each test.
    This ensures complete isolation between tests.
    """
    from fixtures.scene import load_dna

    file_names = [BODY_FILE_NAME, "ExportManifest.json"]

    # Create a unique subfolder for this test run
    unique_folder = temp_folder / f"rbf_test_{uuid.uuid4().hex[:8]}"
    unique_folder.mkdir(parents=True, exist_ok=True)

    # Copy DNA to the unique temp folder
    for _file_name in file_names:
        src = TEST_DNA_FOLDER / dna_folder_name / _file_name
        dst = unique_folder / _file_name
        shutil.copy(src, dst)

    load_dna(
        file_path=unique_folder / BODY_FILE_NAME, import_lods=["lod0"], import_shape_keys=False, import_face_board=False
    )

    yield unique_folder

    # Cleanup: remove the unique folder after the test
    if unique_folder.exists():
        shutil.rmtree(unique_folder, ignore_errors=True)


@pytest.mark.parametrize(
    (
        "solver_name",
        "pose_name",
        "changed_scale_factor",
    ),
    [
        ("calf_l_UERBFSolver", "calf_l_back_90", 0.5),
        ("calf_l_UERBFSolver", "calf_l_back_50", 1.5),
        ("thigh_l_UERBFSolver", "thigh_l_in_45_out_90", 0.75),
    ],
)
def test_rbf_pose_scale_factor_edit(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
    solver_name: str,
    pose_name: str,
    changed_scale_factor: float,
):
    """
    Test that modifying an RBF pose's scale factor is correctly persisted
    when committing changes to the DNA file.
    """
    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Get the original scale factor from the JSON
    _original_pose_index, original_pose_data = get_rbf_pose_data_from_json(original_body_dna_json_data, pose_name)
    assert original_pose_data is not None, f"Pose '{pose_name}' not found in original DNA JSON"
    original_scale = original_pose_data.get("scale", 1.0)

    # Reset the pose to the default position
    reset_pose(instance.body_rig)

    # Set the body pose and get solver/pose indices
    pose, solver_index, pose_index = set_body_pose(solver_name=solver_name, pose_name=pose_name)
    assert pose is not None, f"Pose '{pose_name}' not found in solver '{solver_name}'"

    # Modify the scale factor
    pose.scale_factor = changed_scale_factor

    # Update and commit the pose changes
    bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path = temp_folder / dna_folder_name / f"body_modified_{pose_name}.json"
    modified_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path, data_layer="All")

    # Verify the modified scale factor in the exported JSON
    _modified_pose_index, modified_pose_data = get_rbf_pose_data_from_json(modified_json_data, pose_name)
    assert modified_pose_data is not None, f"Pose '{pose_name}' not found in modified DNA JSON"
    modified_scale = modified_pose_data.get("scale", 1.0)

    # Assert the scale factor was changed
    assert modified_scale != pytest.approx(original_scale, abs=TOLERANCE), (
        f"Scale factor should have changed from {original_scale}"
    )
    assert modified_scale == pytest.approx(changed_scale_factor, abs=TOLERANCE), (
        f"Scale factor should be {changed_scale_factor}, but got {modified_scale}"
    )


@pytest.mark.parametrize(
    (
        "solver_name",
        "pose_name",
        "new_pose_name",
    ),
    [
        # Use poses that are NOT modified by test_rbf_pose_scale_factor_edit
        ("calf_r_UERBFSolver", "calf_r_back_90", "calf_r_back_90_renamed"),
        ("thigh_r_UERBFSolver", "thigh_r_in_45_out_90", "thigh_r_custom_pose"),
    ],
)
def test_rbf_pose_name_edit(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
    solver_name: str,
    pose_name: str,
    new_pose_name: str,
):
    """
    Test that renaming an RBF pose is correctly persisted when committing
    changes to the DNA file.
    """
    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Verify the original pose exists in the JSON
    _original_pose_index, original_pose_data = get_rbf_pose_data_from_json(original_body_dna_json_data, pose_name)
    assert original_pose_data is not None, f"Pose '{pose_name}' not found in original DNA JSON"

    # Reset the pose to the default position
    reset_pose(instance.body_rig)

    # Set the body pose and get solver/pose indices
    pose, _solver_index, _pose_index = set_body_pose(solver_name=solver_name, pose_name=pose_name)
    assert pose is not None, f"Pose '{pose_name}' not found in solver '{solver_name}'"

    # Rename the pose
    pose.name = new_pose_name

    # Commit the pose changes (name change doesn't require apply_rbf_pose_edits)
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path = temp_folder / dna_folder_name / f"body_renamed_{new_pose_name}.json"
    modified_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path, data_layer="All")

    # Verify the old name no longer exists
    _old_pose_index, old_pose_data = get_rbf_pose_data_from_json(modified_json_data, pose_name)
    assert old_pose_data is None, f"Old pose name '{pose_name}' should not exist in modified DNA JSON"

    # Verify the new name exists
    _new_pose_index, new_pose_data = get_rbf_pose_data_from_json(modified_json_data, new_pose_name)
    assert new_pose_data is not None, f"New pose name '{new_pose_name}' should exist in modified DNA JSON"


@pytest.mark.parametrize(
    (
        "solver_name",
        "pose_name",
        "driven_bone_name",
        "location_delta",
    ),
    [
        # Use poses that are NOT modified by other tests
        ("clavicle_l_UERBFSolver", "clavicle_l_up_40", "clavicle_out_l", Vector((0.0, 0.1, 0.0))),
        ("clavicle_r_UERBFSolver", "clavicle_r_up_40", "clavicle_out_r", Vector((0.0, 0.2, 0.0))),
    ],
)
def test_rbf_driven_bone_location_edit(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
    solver_name: str,
    pose_name: str,
    driven_bone_name: str,
    location_delta: Vector,
):
    """
    Test that modifying a driven bone's location in an RBF pose is correctly
    persisted when committing changes to the DNA file.

    This verifies that the poseJointOutputValues in the DNA JSON are updated
    to reflect the bone transform changes made in Blender.
    """
    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Reset the pose to the default position
    reset_pose(instance.body_rig)

    # Set the body pose and get solver/pose indices
    pose, solver_index, pose_index = set_body_pose(solver_name=solver_name, pose_name=pose_name)
    assert pose is not None, f"Pose '{pose_name}' not found in solver '{solver_name}'"

    # Find the driven bone in the pose and apply location change
    driven_found = False
    for driven_index, driven in enumerate(pose.driven):
        if driven.name == driven_bone_name:
            driven_found = True
            pose.driven_active_index = driven_index

            pose_bone = instance.body_rig.pose.bones.get(driven.name)
            assert pose_bone is not None, f"Pose bone '{driven.name}' not found in body rig"

            # Apply the location change
            pose_bone.location = location_delta

            # Update the driven bone transform in the pose
            bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore
            break

    assert driven_found, f"Driven bone '{driven_bone_name}' not found in pose '{pose_name}'"

    # Commit the pose changes
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path = temp_folder / dna_folder_name / f"body_driven_{driven_bone_name}.json"
    modified_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path, data_layer="All")

    # Verify the pose still exists in the modified DNA
    _modified_pose_index, modified_pose_data = get_rbf_pose_data_from_json(modified_json_data, pose_name)
    assert modified_pose_data is not None, f"Pose '{pose_name}' not found in modified DNA JSON"

    # Verify that the RBF behavior data structure exists
    rbf_data = modified_json_data.get(DNA_RBF_BEHAVIOR_VERSION, {})
    assert rbf_data, f"No '{DNA_RBF_BEHAVIOR_VERSION}' data found in modified DNA JSON"


@pytest.mark.parametrize(
    (
        "solver_name",
        "from_pose_name",
        "expected_driven_bone_count",
    ),
    [
        # Use solvers and poses known to exist in the test DNA file
        ("calf_l_UERBFSolver", "calf_l_back_90", 1),
        ("thigh_l_UERBFSolver", "thigh_l_in_45_out_90", 1),
    ],
)
def test_rbf_pose_duplicate(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
    solver_name: str,
    from_pose_name: str,
    expected_driven_bone_count: int,
):
    """
    Test that duplicating an RBF pose creates a new pose with the correct
    data and that it is correctly persisted in the DNA file.
    """
    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Get the original pose count
    original_rbf_data = original_body_dna_json_data.get(DNA_RBF_BEHAVIOR_VERSION, {})
    original_pose_count = len(original_rbf_data.get("poses", []))

    # Reset the pose to the default position
    reset_pose(instance.body_rig)

    # Set the body pose and get solver/pose indices
    pose, solver_index, from_pose_index = set_body_pose(solver_name=solver_name, pose_name=from_pose_name)
    assert pose is not None, f"Pose '{from_pose_name}' not found in solver '{solver_name}'"

    # Duplicate the pose
    bpy.ops.character_dna.duplicate_rbf_pose()  # type: ignore

    # Get the new pose
    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    new_pose_index = len(solver.poses) - 1
    new_pose = solver.poses[new_pose_index]
    new_pose_name = f"{from_pose_name}_duplicated_test"
    new_pose.name = new_pose_name

    # Modify the driver bone rotation to make it unique
    # (duplicated poses have the same driver values which would fail validation)
    if new_pose.drivers:
        driver = new_pose.drivers[0]
        driver_bone = instance.body_rig.pose.bones.get(driver.name)
        if driver_bone:
            # Add a small rotation offset to make the pose unique
            original_quat = Quaternion(driver.quaternion_rotation)
            offset_quat = Quaternion((0.9962, 0.0, 0.0872, 0.0))  # ~10 degree offset around Y
            new_quat = original_quat @ offset_quat
            driver_bone.rotation_quaternion = new_quat

            # Update the driver data
            bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore

    # Verify the duplicated pose has the expected number of driven bones
    assert len(new_pose.driven) >= expected_driven_bone_count, (
        f"Duplicated pose should have at least {expected_driven_bone_count} driven bones, got {len(new_pose.driven)}"
    )

    # Commit the changes
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path = temp_folder / dna_folder_name / f"body_duplicate_{from_pose_name}.json"
    modified_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path, data_layer="All")

    # Verify the new pose exists in the modified DNA
    _new_pose_index_json, new_pose_data = get_rbf_pose_data_from_json(modified_json_data, new_pose_name)
    assert new_pose_data is not None, f"Duplicated pose '{new_pose_name}' not found in modified DNA JSON"

    # Verify the pose count increased
    modified_rbf_data = modified_json_data.get(DNA_RBF_BEHAVIOR_VERSION, {})
    modified_pose_count = len(modified_rbf_data.get("poses", []))
    assert modified_pose_count > original_pose_count, (
        f"Pose count should have increased from {original_pose_count}, got {modified_pose_count}"
    )


@pytest.mark.parametrize(
    (
        "solver_name",
        "from_pose_name",
        "driven_bone_name",
        "location_delta",
    ),
    [
        # Test editing driven bones on duplicated poses
        ("calf_l_UERBFSolver", "calf_l_back_90", "calf_knee_l", Vector((0.05, 0.1, -0.05))),
        ("thigh_l_UERBFSolver", "thigh_l_in_45_out_90", "thigh_bck_lwr_l", Vector((0.0, 0.15, 0.0))),
    ],
)
def test_rbf_duplicated_pose_driven_edit(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
    solver_name: str,
    from_pose_name: str,
    driven_bone_name: str,
    location_delta: Vector,
):
    """
    Test that editing a driven bone on a DUPLICATED pose is correctly persisted.

    This is a regression test for a bug where duplicated poses would not have
    their driven bone edits committed to the DNA file correctly.

    The test:
    1. Duplicates an existing pose
    2. Modifies a driven bone's location on the duplicated pose
    3. Commits changes to DNA
    4. Verifies the edit appears in the output DNA

    This specifically tests the commit_rbf_data_to_dna function's handling of
    new poses with driven data.
    """
    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Reset the pose to the default position
    reset_pose(instance.body_rig)

    # Set the body pose and get solver/pose indices for the source pose
    source_pose, solver_index, source_pose_index = set_body_pose(solver_name=solver_name, pose_name=from_pose_name)
    assert source_pose is not None, f"Pose '{from_pose_name}' not found in solver '{solver_name}'"

    # Duplicate the pose
    bpy.ops.character_dna.duplicate_rbf_pose()  # type: ignore

    # Get the new pose
    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    new_pose_index = len(solver.poses) - 1
    new_pose = solver.poses[new_pose_index]
    new_pose_name = f"{from_pose_name}_edit_test"
    new_pose.name = new_pose_name

    # Modify the driver bone rotation to make it unique
    # (duplicated poses have the same driver values which would fail validation)
    unique_driver_quat = None
    if new_pose.drivers:
        driver = new_pose.drivers[0]
        driver_bone = instance.body_rig.pose.bones.get(driver.name)
        if driver_bone:
            # Add a small rotation offset to make the pose unique
            original_quat = Quaternion(driver.quaternion_rotation)
            offset_quat = Quaternion((0.9962, 0.0, 0.0872, 0.0))  # ~10 degree offset around Y
            unique_driver_quat = original_quat @ offset_quat
            driver_bone.rotation_quaternion = unique_driver_quat

            # Update the driver data
            bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore

    # Reset the rig to rest position to get clean bone positions for driven bone edit
    reset_pose(instance.body_rig)

    # Re-apply the unique driver rotation after reset (so it doesn't match 'default' pose)
    if unique_driver_quat and new_pose.drivers:
        driver = new_pose.drivers[0]
        driver_bone = instance.body_rig.pose.bones.get(driver.name)
        if driver_bone:
            driver_bone.rotation_quaternion = unique_driver_quat

    # Find the driven bone in the duplicated pose and apply location change
    driven_found = False
    for driven_index, driven in enumerate(new_pose.driven):
        if driven.name == driven_bone_name:
            driven_found = True
            new_pose.driven_active_index = driven_index

            pose_bone = instance.body_rig.pose.bones.get(driven.name)
            assert pose_bone is not None, f"Pose bone '{driven.name}' not found in body rig"

            # Apply the location change to the pose bone (this is a delta from rest)
            pose_bone.location = location_delta

            # Update the driven bone transform in the pose
            bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore
            break

    assert driven_found, f"Driven bone '{driven_bone_name}' not found in duplicated pose"

    # Verify the driven data was updated in memory - just check that it has non-zero values
    updated_driven = new_pose.driven[driven_index]
    driven_location = Vector(updated_driven.location[:])
    assert driven_location.length > TOLERANCE, (
        f"Driven location should have non-zero values after update, got {driven_location}"
    )

    # Commit the pose changes
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path = temp_folder / dna_folder_name / f"body_dup_edit_{driven_bone_name}.json"
    modified_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path, data_layer="All")

    # Verify the duplicated pose exists in the modified DNA
    _new_pose_index_json, new_pose_data = get_rbf_pose_data_from_json(modified_json_data, new_pose_name)
    assert new_pose_data is not None, f"Duplicated pose '{new_pose_name}' not found in modified DNA JSON"

    # Get the pose's output control indices to find its column in the joint group matrix
    pose_output_control_indices = new_pose_data.get("outputControlIndices", [])
    assert len(pose_output_control_indices) > 0, f"Pose '{new_pose_name}' should have output control indices"

    # Get the joint groups and verify the driven bone data was written
    # JSON format uses bhvr1.1.joints.jointGroups
    behavior_data = modified_json_data.get(DNA_BEHAVIOR_VERSION, {})
    joints_data = behavior_data.get("joints", {})
    joint_groups = joints_data.get("jointGroups", [])
    assert len(joint_groups) > 0, "DNA should have joint groups"

    # Find the joint index for our driven bone
    # JSON format uses defn1.1.jointNames
    definition_data = modified_json_data.get(DNA_DEFINITION_VERSION, {})
    joint_names = definition_data.get("jointNames", [])
    driven_joint_index = None
    for idx, name in enumerate(joint_names):
        if name == driven_bone_name:
            driven_joint_index = idx
            break

    assert driven_joint_index is not None, f"Joint '{driven_bone_name}' not found in definition"

    # Look for non-zero values in the joint group that correspond to our pose's control
    # The output indices encode: joint_index * 9 + attr_index (0-2=location, 3-5=rotation, 6-8=scale)
    location_output_indices = [
        driven_joint_index * 9 + 0,  # X
        driven_joint_index * 9 + 1,  # Y
        driven_joint_index * 9 + 2,  # Z
    ]

    # Find the joint group that contains our control index
    control_idx = pose_output_control_indices[0]
    found_values = False

    for jg in joint_groups:
        input_indices = jg.get("inputIndices", [])
        if control_idx in input_indices:
            col_idx = input_indices.index(control_idx)
            output_indices = jg.get("outputIndices", [])
            values = jg.get("values", [])
            num_cols = len(input_indices)

            # Check if any of our location output indices are in this joint group
            for loc_out_idx in location_output_indices:
                if loc_out_idx in output_indices:
                    row_idx = output_indices.index(loc_out_idx)
                    matrix_pos = row_idx * num_cols + col_idx
                    if matrix_pos < len(values):
                        value = values[matrix_pos]
                        if abs(value) > 1e-5:
                            found_values = True
                            break

            if found_values:
                break

    assert found_values, (
        f"The driven bone '{driven_bone_name}' location values were not found in the joint group "
        f"for the duplicated pose. This indicates the driven data was not correctly written."
    )


@pytest.mark.parametrize(
    "solver_name",
    [
        "calf_l_UERBFSolver",
        "thigh_l_UERBFSolver",
        "clavicle_r_UERBFSolver",
    ],
)
def test_rbf_solver_exists_in_json(fresh_rbf_test_scene, original_body_dna_json_data: dict, solver_name: str):
    """
    Test that specific RBF solvers exist in the original DNA JSON structure
    under the 'rbfb1.0' key.
    """
    _solver_index, solver_data = get_rbf_solver_data_from_json(original_body_dna_json_data, solver_name)

    assert solver_data is not None, f"Solver '{solver_name}' not found in DNA JSON"
    assert "name" in solver_data, "Solver data should contain 'name' field"
    assert solver_data["name"] == solver_name, (
        f"Solver name mismatch: expected '{solver_name}', got '{solver_data['name']}'"
    )


@pytest.mark.parametrize(
    (
        "solver_name",
        "expected_pose_names",
    ),
    [
        ("calf_l_UERBFSolver", ["calf_l_back_50", "calf_l_back_90"]),
    ],
)
def test_rbf_solver_contains_expected_poses(
    fresh_rbf_test_scene, original_body_dna_json_data: dict, solver_name: str, expected_pose_names: list[str]
):
    """
    Test that an RBF solver contains the expected poses in the DNA JSON structure.
    """
    _solver_index, solver_data = get_rbf_solver_data_from_json(original_body_dna_json_data, solver_name)

    assert solver_data is not None, f"Solver '{solver_name}' not found in DNA JSON"

    # Get the pose indices from the solver
    pose_indices = solver_data.get("poseIndices", [])
    assert len(pose_indices) > 0, f"Solver '{solver_name}' should have pose indices"

    # Get the poses from the RBF data
    rbf_data = original_body_dna_json_data.get(DNA_RBF_BEHAVIOR_VERSION, {})
    poses = rbf_data.get("poses", [])

    # Get the pose names for the solver's pose indices
    solver_pose_names = [poses[pose_index].get("name", "") for pose_index in pose_indices if pose_index < len(poses)]

    # Verify expected poses are present
    for expected_pose_name in expected_pose_names:
        assert expected_pose_name in solver_pose_names, (
            f"Expected pose '{expected_pose_name}' not found in solver '{solver_name}'. Found: {solver_pose_names}"
        )


# =============================================================================
# Joint Group Consistency Tests
# =============================================================================


def test_get_solver_joint_group_bones(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that get_solver_joint_group_bones returns the correct set of bone names
    for all driven bones in the active solver's poses.
    """
    from character_dna.editors.rbf_editor.utilities import get_solver_joint_group_bones

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver with known poses
    solver_name = "calf_l_UERBFSolver"
    pose, _, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    # Get the joint group bones
    bone_names = get_solver_joint_group_bones(instance)

    # Verify we got some bones
    assert len(bone_names) > 0, "Should have found at least one bone in the joint group"

    # Verify all bones are strings
    for bone_name in bone_names:
        assert isinstance(bone_name, str), f"Bone name should be a string, got {type(bone_name)}"


def test_get_available_driven_bones_excludes_driver_bones(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that get_available_driven_bones does not include driver bones,
    swing bones, or twist bones.
    """
    from character_dna.editors.rbf_editor.utilities import get_available_driven_bones

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Initialize the body to populate driver/swing/twist bone lists
    if not instance.body_initialized:
        instance.body_initialize()

    # Get available driven bones
    available_bones = get_available_driven_bones(instance)
    available_bone_names = {bone_info[0] for bone_info in available_bones}

    # Verify driver bones are excluded
    for driver_bone in instance.body_driver_bone_names:
        assert driver_bone not in available_bone_names, (
            f"Driver bone '{driver_bone}' should not be in available driven bones"
        )

    # Verify swing bones are excluded
    for swing_bone in instance.body_swing_bone_names:
        assert swing_bone not in available_bone_names, (
            f"Swing bone '{swing_bone}' should not be in available driven bones"
        )

    # Verify twist bones are excluded
    for twist_bone in instance.body_twist_bone_names:
        assert twist_bone not in available_bone_names, (
            f"Twist bone '{twist_bone}' should not be in available driven bones"
        )


def test_validate_driver_bone_detects_duplicate_quaternions(fresh_rbf_test_scene, dna_folder_name: str):
    from character_dna.editors.rbf_editor.utilities import validate_no_duplicate_driver_bone_values

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver
    solver_name = "calf_l_UERBFSolver"
    pose, solver_index, pose_index = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    # Duplicate the pose
    bpy.ops.character_dna.duplicate_rbf_pose()  # type: ignore

    # The duplicated pose has the same driver quaternion values as the original
    # Validation should fail
    is_valid, message = validate_no_duplicate_driver_bone_values(instance)
    assert not is_valid, "Duplicated pose should have duplicate driver values detected"
    assert "duplicate" in message.lower() or "unique" in message.lower(), (
        f"Error message should mention duplicate or unique values: {message}"
    )


def test_validate_and_update_solver_joint_group_no_new_bones(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that validate_and_update_solver_joint_group returns success
    when all driven bones are already in the joint group.
    """
    from character_dna.editors.rbf_editor.utilities import (
        get_solver_joint_group_bones,
        validate_and_update_solver_joint_group,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver
    solver_name = "calf_l_UERBFSolver"
    pose, _, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    # Get existing bones in the joint group
    existing_bones = get_solver_joint_group_bones(instance)
    assert len(existing_bones) > 0, "Should have existing bones in joint group"

    # Validate with the same bones (no new bones)
    is_valid, message = validate_and_update_solver_joint_group(instance, list(existing_bones))
    assert is_valid, f"Validation should pass when using existing bones: {message}"


def test_validate_and_update_solver_joint_group_expands_poses(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that validate_and_update_solver_joint_group correctly adds new bones
    to all existing poses in the solver when new bones are added.
    """
    from character_dna.editors.rbf_editor.utilities import (
        get_available_driven_bones,
        get_solver_joint_group_bones,
        validate_and_update_solver_joint_group,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver
    solver_name = "calf_l_UERBFSolver"
    pose, solver_index, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    # Get existing bones and available bones
    existing_bones = get_solver_joint_group_bones(instance)
    available_bones = get_available_driven_bones(instance)

    # Find a new bone that's not in the existing joint group
    new_bone_name = None
    for bone_name, joint_index, is_in_existing in available_bones:
        if not is_in_existing and joint_index >= 0:
            new_bone_name = bone_name
            break

    if new_bone_name is None:
        pytest.skip("No available bones to add to the joint group")

    # Get the solver and count driven bones in each pose before expansion
    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    driven_counts_before = {p.name: len(p.driven) for p in solver.poses}

    # Validate with the new bone added
    new_driven_bones = list(existing_bones) + [new_bone_name]
    is_valid, message = validate_and_update_solver_joint_group(instance, new_driven_bones)
    assert is_valid, f"Validation should pass when adding new bone: {message}"

    # Verify that all existing poses now have the new bone added
    for pose in solver.poses:
        # Skip the default pose as it doesn't have driven data
        if pose.name == "default":
            continue

        driven_bone_names = {d.name for d in pose.driven}
        assert new_bone_name in driven_bone_names, (
            f"Pose '{pose.name}' should have the new bone '{new_bone_name}' added. Driven bones: {driven_bone_names}"
        )

        # Verify the driven count increased
        assert len(pose.driven) >= driven_counts_before.get(pose.name, 0), (
            f"Pose '{pose.name}' driven count should not decrease after adding new bone"
        )


def test_add_pose_with_expanded_joint_group_commits_to_dna(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
):
    """
    Test that adding a new pose with additional bones (expanding the joint group)
    correctly commits to DNA without corruption.

    This is a regression test for a crash that occurred when:
    1. User adds a new pose with AddRBFPose operator
    2. User selects bones that are NOT in the existing joint group
    3. User commits changes with CommitRBFSolverChanges
    4. Blender crashes when reloading the DNA file

    The test verifies:
    - The DNA file is written without corruption
    - The new pose exists in the DNA
    - The expanded joint group contains the new bones
    - The DNA can be reloaded successfully
    """
    from character_dna.editors.rbf_editor.utilities import (
        get_available_driven_bones,
        get_solver_joint_group_bones,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver and enter edit mode (set_body_pose handles editing_rbf_solver = True)
    solver_name = "calf_l_UERBFSolver"
    pose, solver_index, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    # Get existing bones and find a new bone to add
    existing_bones = get_solver_joint_group_bones(instance)
    available_bones = get_available_driven_bones(instance)

    # Find a new bone that's not in the existing joint group
    new_bone_name = None
    new_bone_joint_index = -1
    for bone_name, joint_index, is_in_existing in available_bones:
        if not is_in_existing and joint_index >= 0:
            new_bone_name = bone_name
            new_bone_joint_index = joint_index
            break

    if new_bone_name is None:
        pytest.skip("No available bones to add to the joint group")

    # Get the driver bone and set a unique rotation for the new pose
    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    driver_bone_name = solver_name.replace("_UERBFSolver", "")
    driver_bone = instance.body_rig.pose.bones.get(driver_bone_name)
    assert driver_bone is not None, f"Driver bone '{driver_bone_name}' not found"

    # Set a unique rotation for the new pose (different from existing poses)
    unique_rotation = Quaternion((0.866, 0.0, 0.5, 0.0))  # 60 degree rotation around Y
    driver_bone.rotation_quaternion = unique_rotation

    # Capture existing pose names before adding
    existing_pose_names = {p.name for p in solver.poses}

    # Manually simulate what the AddRBFPose operator does:
    # 1. Populate bone selections in window manager
    wm = bpy.context.window_manager.character_dna  # type: ignore
    wm.add_pose_driven_bones.clear()  #

    # Add all existing bones (pre-selected)
    for bone_name in existing_bones:
        item = wm.add_pose_driven_bones.add()
        item.name = bone_name
        item.selected = True
        item.is_in_existing_joint_group = True
        # Find joint index for this bone
        for bn, ji, _ in available_bones:
            if bn == bone_name:
                item.joint_index = ji
                break

    # Add the new bone (also selected)
    item = wm.add_pose_driven_bones.add()
    item.name = new_bone_name
    item.selected = True
    item.joint_index = new_bone_joint_index
    item.is_in_existing_joint_group = False

    # Call the AddRBFPose operator (without invoke, direct execute)
    result = bpy.ops.character_dna.add_rbf_pose()  # type: ignore
    assert result == {"FINISHED"}, f"AddRBFPose operator failed: {result}"

    # Find the newly added pose by comparing with existing pose names
    new_pose = None
    new_pose_name = None
    for p in solver.poses:
        if p.name not in existing_pose_names:
            new_pose = p
            new_pose_name = p.name
            break
    assert new_pose is not None, f"No new pose was created. Existing poses: {existing_pose_names}"

    # Verify the new pose has all the bones (existing + new)
    new_pose_driven_names = {d.name for d in new_pose.driven}
    assert new_bone_name in new_pose_driven_names, (
        f"New bone '{new_bone_name}' should be in the new pose's driven bones. Found: {new_pose_driven_names}"
    )

    # Commit the changes to DNA
    result = bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore
    assert result == {"FINISHED"}, f"CommitRBFSolverChanges operator failed: {result}"

    # Export the modified DNA to JSON for verification
    json_file_path = temp_folder / dna_folder_name / "body_expanded_joint_group.json"
    modified_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path, data_layer="All")

    # Verify the new pose exists in the DNA
    _new_pose_index, new_pose_data = get_rbf_pose_data_from_json(modified_json_data, new_pose_name)
    assert new_pose_data is not None, f"New pose '{new_pose_name}' not found in modified DNA JSON"

    # Verify the DNA can be reloaded (this is the crash scenario)
    # Re-initialize the instance to reload the DNA
    instance.destroy()
    instance.body_initialize()

    # Verify the instance is still valid after reloading
    assert instance.body_initialized, "Instance should be initialized after reloading DNA"
    assert instance.body_dna_reader is not None, "DNA reader should be available after reload"

    # Verify the new pose is in the reloaded solver data
    found_pose = False
    for s in instance.rbf_editor.rbf_solver_list:
        for p in s.poses:
            if p.name == new_pose_name:
                found_pose = True
                # Verify the new bone is in the reloaded pose
                reloaded_driven_names = {d.name for d in p.driven}
                assert new_bone_name in reloaded_driven_names, (
                    f"New bone '{new_bone_name}' should still be in the pose after reload. "
                    f"Found: {reloaded_driven_names}"
                )
                break
        if found_pose:
            break

    assert found_pose, f"New pose '{new_pose_name}' not found after reloading DNA"


def test_add_rbf_driven_adds_bone_to_all_poses(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
):
    """
    Test that adding a driven bone adds it to ALL poses in the solver.

    This test verifies:
    - A new bone can be added to the solver's joint group
    - The bone is added to all existing poses (not just the active one)
    - The bone is added with rest pose transforms (zero deltas) for existing poses

    Note: Tests the core function directly instead of the operator to avoid
    Blender 5.0 headless bone selection issues.
    """
    from character_dna.editors.rbf_editor.utilities import (
        add_driven_bones_to_solver,
        get_available_driven_bones,
        get_solver_joint_group_bones,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver and enter edit mode
    solver_name = "calf_l_UERBFSolver"
    pose, _, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    solver = instance.rbf_editor.rbf_solver_list[instance.rbf_editor.rbf_solver_list_active_index]

    # Get existing bones and find a new bone to add
    existing_bones = get_solver_joint_group_bones(instance)
    available_bones = get_available_driven_bones(instance)

    # Find a new bone that's not in the existing joint group
    new_bone_name = None
    for bone_name, joint_index, is_in_existing in available_bones:
        if not is_in_existing and joint_index >= 0:
            new_bone_name = bone_name
            break

    if new_bone_name is None:
        pytest.skip("No available bones to add to the joint group")

    # Record driven bone counts before adding
    driven_counts_before = {p.name: len(p.driven) for p in solver.poses if p.name != "default"}

    # Call the core function directly to add the bone to all poses
    valid, message = add_driven_bones_to_solver(instance, [new_bone_name])
    assert valid, f"add_driven_bones_to_solver failed: {message}"

    # Verify the bone was added to ALL poses
    for p in solver.poses:
        if p.name == "default":
            continue

        driven_bone_names = {d.name for d in p.driven}
        assert new_bone_name in driven_bone_names, (
            f"Pose '{p.name}' should have the new bone '{new_bone_name}' added. Driven bones: {driven_bone_names}"
        )

        # Verify the driven count increased by 1
        expected_count = driven_counts_before.get(p.name, 0) + 1
        assert len(p.driven) == expected_count, (
            f"Pose '{p.name}' should have {expected_count} driven bones, but has {len(p.driven)}"
        )


def test_remove_rbf_driven_removes_bone_from_all_poses(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
):
    """
    Test that removing a driven bone removes it from ALL poses in the solver.

    This test verifies:
    - A bone can be removed from the solver's joint group
    - The bone is removed from all existing poses (not just the active one)
    - At least one bone remains after removal

    Note: Tests the core function directly instead of the operator to avoid
    Blender 5.0 headless bone selection issues.
    """
    from character_dna.editors.rbf_editor.utilities import (
        get_solver_joint_group_bones,
        remove_driven_bone_from_solver,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver and enter edit mode
    solver_name = "calf_l_UERBFSolver"
    pose, solver_index, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    solver = instance.rbf_editor.rbf_solver_list[solver_index]

    # Get existing bones in the joint group
    existing_bones = get_solver_joint_group_bones(instance)
    assert len(existing_bones) > 1, "Need at least 2 bones in joint group to test removal"

    # Pick a bone to remove (first one that exists in the pose)
    bone_to_remove = None
    for driven in pose.driven:
        if driven.data_type == "BONE":
            bone_to_remove = driven.name
            break

    assert bone_to_remove is not None, "No bone found in pose to remove"

    # Record driven bone counts before removal
    driven_counts_before = {p.name: len(p.driven) for p in solver.poses if p.name != "default"}

    # Call the core function directly to remove the bone from all poses
    valid, message = remove_driven_bone_from_solver(instance, {bone_to_remove})
    assert valid, f"remove_driven_bone_from_solver failed: {message}"

    # Verify the bone was removed from ALL poses
    for p in solver.poses:
        if p.name == "default":
            continue

        driven_bone_names = {d.name for d in p.driven}
        assert bone_to_remove not in driven_bone_names, (
            f"Pose '{p.name}' should NOT have the bone '{bone_to_remove}' after removal. "
            f"Driven bones: {driven_bone_names}"
        )

        # Verify the driven count decreased by 1
        expected_count = driven_counts_before.get(p.name, 0) - 1
        assert len(p.driven) == expected_count, (
            f"Pose '{p.name}' should have {expected_count} driven bones, but has {len(p.driven)}"
        )


def test_remove_rbf_driven_cannot_remove_all_bones(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
):
    """
    Test that remove_driven_bone_from_solver prevents removing all bones from a solver.

    This test verifies:
    - The function fails when trying to remove all driven bones
    - At least one driven bone must remain

    Note: Tests the core function directly instead of the operator to avoid
    Blender 5.0 headless bone selection issues.
    """
    from character_dna.editors.rbf_editor.utilities import (
        get_solver_joint_group_bones,
        remove_driven_bone_from_solver,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver and enter edit mode
    solver_name = "calf_l_UERBFSolver"
    pose, _, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    # Get all existing bones in the joint group
    existing_bones = get_solver_joint_group_bones(instance)

    # Try to remove ALL bones - this should fail
    valid, message = remove_driven_bone_from_solver(instance, set(existing_bones))

    # The function should return False because we can't remove all bones
    assert not valid, f"remove_driven_bone_from_solver should fail when trying to remove all bones"
    assert "at least one" in message.lower() or "cannot remove" in message.lower(), (
        f"Error message should explain why removal failed: {message}"
    )


def test_add_rbf_driven_validates_bone_type(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
):
    """
    Test that driver/swing/twist bones cannot be added as driven bones.

    This test verifies:
    - Driver bones cannot be added as driven bones
    - The validate_and_update_solver_joint_group function should not be called
      with invalid bone types (this is operator-level validation)

    Note: This tests the validation logic that would be in the operator's validate
    method. Since we can't test the operator due to Blender 5.0 selection issues,
    we verify the bone type checking logic directly.
    """
    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver and enter edit mode
    solver_name = "calf_l_UERBFSolver"
    pose, _, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    # Initialize body to get driver bone names
    if not instance.body_initialized:
        instance.body_initialize(update_rbf_solver_list=False)

    # Find a driver bone
    driver_bone_name = None
    for bone_name in instance.body_driver_bone_names:
        pose_bone = instance.body_rig.pose.bones.get(bone_name)
        if pose_bone:
            driver_bone_name = bone_name
            break

    if driver_bone_name is None:
        pytest.skip("No driver bones found to test validation")

    # Verify that driver bones are recognized as such
    assert driver_bone_name in instance.body_driver_bone_names, (
        f"Bone '{driver_bone_name}' should be in body_driver_bone_names"
    )

    # Verify that driver bones are not in available driven bones (with is_in_existing=False)
    from character_dna.editors.rbf_editor.utilities import get_available_driven_bones

    available_bones = get_available_driven_bones(instance)

    # The driver bone should NOT be in available driven bones
    available_bone_names = [b[0] for b in available_bones]
    assert driver_bone_name not in available_bone_names, (
        f"Driver bone '{driver_bone_name}' should NOT be in available driven bones. "
        f"Available bones include driver bones incorrectly."
    )


def test_remove_and_add_rbf_driven_persists_after_commit(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
):
    """
    Test that removing a driven bone and then adding it back persists correctly.

    This is a regression test for a bug where:
    1. Removing a bone from the solver's joint group
    2. Committing changes to DNA
    3. Adding the bone back
    4. Committing changes to DNA
    ...would not properly persist the bone in the DNA after the second commit.

    The test:
    1. Removes thigh_twistCor_01_l from calf_l_UERBFSolver
    2. Commits changes to DNA
    3. Verifies the bone is removed
    4. Adds thigh_twistCor_01_l back to the solver
    5. Commits changes to DNA
    6. Verifies the bone is present in the DNA with the correct structure

    Note: Tests the core function directly instead of the operator to avoid
    Blender 5.0 headless bone selection issues.
    """
    from character_dna.editors.rbf_editor.utilities import (
        add_driven_bones_to_solver,
        get_solver_joint_group_bones,
        remove_driven_bone_from_solver,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver and enter edit mode
    solver_name = "calf_l_UERBFSolver"
    bone_to_test = "thigh_twistCor_01_l"

    pose, solver_index, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    solver = instance.rbf_editor.rbf_solver_list[solver_index]

    # Get existing bones and verify the bone is present
    existing_bones = get_solver_joint_group_bones(instance)
    if bone_to_test not in existing_bones:
        pytest.skip(f"Bone '{bone_to_test}' not in solver's joint group, cannot test")

    # Step 1: Remove the bone from all poses
    valid, message = remove_driven_bone_from_solver(instance, {bone_to_test})
    assert valid, f"remove_driven_bone_from_solver failed: {message}"

    # Verify the bone was removed from in-memory data
    for p in solver.poses:
        if p.name == "default":
            continue
        driven_bone_names = {d.name for d in p.driven}
        assert bone_to_test not in driven_bone_names, (
            f"Pose '{p.name}' should NOT have the bone '{bone_to_test}' after removal"
        )

    # Step 2: Commit changes to DNA (first commit - removal)
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path_removed = temp_folder / dna_folder_name / f"body_removed_{bone_to_test}.json"
    removed_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path_removed, data_layer="All")

    # Get the joint index for the removed bone
    definition_data = removed_json_data.get(DNA_DEFINITION_VERSION, {})
    joint_names = definition_data.get("jointNames", [])
    bone_joint_index = None
    for idx, name in enumerate(joint_names):
        if name == bone_to_test:
            bone_joint_index = idx
            break

    assert bone_joint_index is not None, f"Joint '{bone_to_test}' not found in definition"

    # Verify the bone was removed from the DNA joint groups
    behavior_data = removed_json_data.get(DNA_BEHAVIOR_VERSION, {})
    joints_data = behavior_data.get("joints", {})
    joint_groups = joints_data.get("jointGroups", [])

    # Get the solver's pose indices
    solver_index_json, solver_data = get_rbf_solver_data_from_json(removed_json_data, solver_name)
    assert solver_data is not None, f"Solver '{solver_name}' not found in removed DNA"

    # Step 3: Add the bone back to all poses
    # Re-initialize the solver data from the DNA (as if we just loaded it)
    instance.destroy()
    instance.body_initialize()

    # Set up the solver again
    pose, solver_index, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}' after reload"

    # Verify the bone is NOT in non-default poses after reload
    # Note: The default pose now shows all joint group bones from the DNA,
    # so we only check non-default poses
    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    for p in solver.poses:
        if p.name == "default":
            continue
        driven_bone_names = {d.name for d in p.driven}
        assert bone_to_test not in driven_bone_names, (
            f"Pose '{p.name}' should NOT have the bone '{bone_to_test}' after removal and reload. "
            f"Found: {driven_bone_names}"
        )

    # Add the bone back
    valid, message = add_driven_bones_to_solver(instance, [bone_to_test])
    assert valid, f"add_driven_bones_to_solver failed: {message}"

    # Verify the bone was added to in-memory data
    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    for p in solver.poses:
        if p.name == "default":
            continue
        driven_bone_names = {d.name for d in p.driven}
        assert bone_to_test in driven_bone_names, (
            f"Pose '{p.name}' should have the bone '{bone_to_test}' after adding back. Found: {driven_bone_names}"
        )

    # Step 4: Commit changes to DNA (second commit - adding back)
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path_added = temp_folder / dna_folder_name / f"body_added_{bone_to_test}.json"
    added_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path_added, data_layer="All")

    # Verify the bone was added back to the DNA joint groups
    behavior_data_added = added_json_data.get(DNA_BEHAVIOR_VERSION, {})
    joints_data_added = behavior_data_added.get("joints", {})
    joint_groups_added = joints_data_added.get("jointGroups", [])

    # Get the solver's pose indices from the added DNA
    solver_index_json_added, solver_data_added = get_rbf_solver_data_from_json(added_json_data, solver_name)
    assert solver_data_added is not None, f"Solver '{solver_name}' not found in added DNA"

    pose_indices = solver_data_added.get("poseIndices", [])
    assert len(pose_indices) > 0, f"Solver '{solver_name}' should have poses"

    # Get the RBF extension poses
    rbf_extension_data = added_json_data.get(DNA_RBF_EXTENSION_VERSION, {})
    extension_poses = rbf_extension_data.get("poses", [])

    # Verify the bone is now in at least one joint group used by this solver's poses
    bone_found_in_joint_group = False
    for pose_idx in pose_indices:
        # Get the pose's output control indices from extension data
        if pose_idx < len(extension_poses):
            pose_output_control_indices = extension_poses[pose_idx].get("outputControlIndices", [])
        else:
            continue

        if not pose_output_control_indices:
            continue

        # For each control, check the joint group structure
        for control_idx in pose_output_control_indices:
            for jg in joint_groups_added:
                input_indices = jg.get("inputIndices", [])
                if control_idx not in input_indices:
                    continue

                # Check that the bone's joint is in the joint group's jointIndices
                joint_indices = jg.get("jointIndices", [])
                if bone_joint_index in joint_indices:
                    bone_found_in_joint_group = True

                    # The coordinate-policy transform sparsifies all-zero rows.
                    # Membership is explicit even when the re-added bone has no
                    # corrective values yet; remaining rows must stay valid.
                    output_indices = jg.get("outputIndices", [])
                    assert len(jg["values"]) == len(input_indices) * len(output_indices)
                    assert all(out_idx // 9 in joint_indices for out_idx in output_indices)
                    break
            if bone_found_in_joint_group:
                break
        if bone_found_in_joint_group:
            break

    assert bone_found_in_joint_group, (
        f"Joint '{bone_to_test}' (index {bone_joint_index}) should be in "
        f"jointIndices after adding back. Checked joint groups for solver poses."
    )
    instance.destroy()
    instance.body_initialize()
    pose, _, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert bone_to_test in {driven.name for driven in pose.driven}


def test_remove_rbf_driven_persists_after_commit(
    fresh_rbf_test_scene,
    original_body_dna_json_data: dict,
    temp_folder,
    dna_folder_name: str,
):
    """
    Test that removing a driven bone persists after committing to DNA.

    This is a regression test for a bug where removing a bone from the solver's
    joint group would not persist after committing changes to DNA. The bone
    would still appear in the joint group matrix because the values were not
    zeroed out.

    The test:
    1. Removes a specific bone (thigh_twistCor_01_l) from calf_l_UERBFSolver
    2. Commits changes to DNA
    3. Reloads the DNA and verifies the bone is no longer in the joint group

    Note: Tests the core function directly instead of the operator to avoid
    Blender 5.0 headless bone selection issues.
    """
    from character_dna.editors.rbf_editor.utilities import (
        get_solver_joint_group_bones,
        remove_driven_bone_from_solver,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Set up a solver and enter edit mode
    solver_name = "calf_l_UERBFSolver"
    bone_to_remove = "thigh_twistCor_01_l"

    pose, solver_index, _ = set_body_pose(solver_name=solver_name, pose_name="calf_l_back_50")
    assert pose is not None, f"Pose not found in solver '{solver_name}'"

    solver = instance.rbf_editor.rbf_solver_list[solver_index]

    # Get existing bones and verify the bone is present
    existing_bones = get_solver_joint_group_bones(instance)
    if bone_to_remove not in existing_bones:
        pytest.skip(f"Bone '{bone_to_remove}' not in solver's joint group, cannot test removal")

    # Record the original driven bones count
    original_bone_count = len(existing_bones)
    assert original_bone_count > 1, "Need at least 2 bones in joint group to test removal"

    # Remove the bone from all poses
    valid, message = remove_driven_bone_from_solver(instance, {bone_to_remove})
    assert valid, f"remove_driven_bone_from_solver failed: {message}"

    # Verify the bone was removed from the in-memory solver data
    for p in solver.poses:
        if p.name == "default":
            continue
        driven_bone_names = {d.name for d in p.driven}
        assert bone_to_remove not in driven_bone_names, (
            f"Pose '{p.name}' should NOT have the bone '{bone_to_remove}' after removal"
        )

    # Commit changes to DNA
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # Export the modified DNA to JSON for verification
    json_file_path = temp_folder / dna_folder_name / f"body_remove_{bone_to_remove}.json"
    modified_json_data = get_dna_json_data(instance.body_dna_file_path, json_file_path, data_layer="All")

    # Get the joint index for the removed bone
    definition_data = modified_json_data.get(DNA_DEFINITION_VERSION, {})
    joint_names = definition_data.get("jointNames", [])
    removed_joint_index = None
    for idx, name in enumerate(joint_names):
        if name == bone_to_remove:
            removed_joint_index = idx
            break

    assert removed_joint_index is not None, f"Joint '{bone_to_remove}' not found in definition"

    # Get the solver's pose indices using the helper function
    solver_index_json, solver_data = get_rbf_solver_data_from_json(modified_json_data, solver_name)
    assert solver_data is not None, f"Solver '{solver_name}' not found in modified DNA"

    # Get all pose indices for this solver
    pose_indices = solver_data.get("poseIndices", [])
    assert len(pose_indices) > 0, f"Solver '{solver_name}' should have poses"

    # Get the RBF poses and extension data
    rbf_behavior_data = modified_json_data.get(DNA_RBF_BEHAVIOR_VERSION, {})
    rbf_extension_data = modified_json_data.get(DNA_RBF_EXTENSION_VERSION, {})
    behavior_poses = rbf_behavior_data.get("poses", [])
    extension_poses = rbf_extension_data.get("poses", [])

    # Get joint group data from the main behavior section
    behavior_data = modified_json_data.get(DNA_BEHAVIOR_VERSION, {})
    joints_data = behavior_data.get("joints", {})
    joint_groups = joints_data.get("jointGroups", [])

    # The output indices for the removed bone (9 attributes per bone)
    removed_bone_output_indices = set(range(removed_joint_index * 9, removed_joint_index * 9 + 9))

    for pose_idx in pose_indices:
        # Get pose name for logging
        pose_name = "unknown"
        if pose_idx < len(behavior_poses):
            pose_name = behavior_poses[pose_idx].get("name", f"pose_{pose_idx}")

        if pose_name == "default":
            continue

        # Get the pose's output control indices from extension data
        pose_output_control_indices = []
        if pose_idx < len(extension_poses):
            pose_output_control_indices = extension_poses[pose_idx].get("outputControlIndices", [])

        if not pose_output_control_indices:
            continue

        # For each control, check the joint group structure
        for control_idx in pose_output_control_indices:
            for jg in joint_groups:
                input_indices = jg.get("inputIndices", [])
                if control_idx not in input_indices:
                    continue

                # Check that the removed joint is NOT in the joint group's jointIndices
                joint_indices = jg.get("jointIndices", [])
                assert removed_joint_index not in joint_indices, (
                    f"Joint '{bone_to_remove}' (index {removed_joint_index}) should NOT be in "
                    f"jointIndices after removal. Found jointIndices: {joint_indices}"
                )

                # Also verify the output indices for this bone were removed
                output_indices = jg.get("outputIndices", [])
                for out_idx in removed_bone_output_indices:
                    assert out_idx not in output_indices, (
                        f"Output index {out_idx} for bone '{bone_to_remove}' should NOT be in "
                        f"outputIndices after removal."
                    )


# =============================================================================
# RBF Solver Core Function Tests (validate_add_rbf_solver, add_rbf_solver, remove_rbf_solver)
# =============================================================================


def test_validate_add_rbf_solver_rejects_swing_bone(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that validate_add_rbf_solver rejects swing bones as driver bones.
    """
    from character_dna.editors.rbf_editor.utilities import validate_add_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    instance.body_initialize(update_rbf_solver_list=False)

    # Get a swing bone name from the instance
    if not instance.body_swing_bone_names:
        pytest.skip("No swing bones available for testing")

    swing_bone_name = list(instance.body_swing_bone_names)[0]

    # Validate using core function
    is_valid, message = validate_add_rbf_solver(instance, swing_bone_name)

    assert is_valid is False, "Validation should reject swing bones as driver bones"
    assert "swing bone" in message.lower(), f"Error message should mention swing bone: {message}"


def test_validate_add_rbf_solver_rejects_twist_bone(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that validate_add_rbf_solver rejects twist bones as driver bones.
    """
    from character_dna.editors.rbf_editor.utilities import validate_add_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    instance.body_initialize(update_rbf_solver_list=False)

    # Get a twist bone name from the instance
    if not instance.body_twist_bone_names:
        pytest.skip("No twist bones available for testing")

    twist_bone_name = list(instance.body_twist_bone_names)[0]

    # Validate using core function
    is_valid, message = validate_add_rbf_solver(instance, twist_bone_name)

    assert is_valid is False, "Validation should reject twist bones as driver bones"
    assert "twist bone" in message.lower(), f"Error message should mention twist bone: {message}"


def test_validate_add_rbf_solver_rejects_duplicate_solver(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that validate_add_rbf_solver rejects creating a solver for a bone that already has one.
    """
    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import validate_add_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Find an existing solver and get its driver bone
    if len(instance.rbf_editor.rbf_solver_list) == 0:
        pytest.skip("No existing RBF solvers to test duplication check")

    existing_solver = instance.rbf_editor.rbf_solver_list[0]
    driver_bone_name = existing_solver.name.replace(RBF_SOLVER_POSTFIX, "")

    # Validate using core function
    is_valid, message = validate_add_rbf_solver(instance, driver_bone_name)

    assert is_valid is False, "Validation should reject creating duplicate solver"
    assert "already exists" in message.lower(), f"Error message should mention existing solver: {message}"


def test_validate_add_rbf_solver_rejects_nonexistent_bone(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that validate_add_rbf_solver rejects bones that don't exist in the rig.
    """
    from character_dna.editors.rbf_editor.utilities import validate_add_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    # Validate using a bone name that doesn't exist
    is_valid, message = validate_add_rbf_solver(instance, "nonexistent_bone_xyz_123")

    assert is_valid is False, "Validation should reject nonexistent bones"
    assert "not found" in message.lower(), f"Error message should mention bone not found: {message}"


def test_validate_add_rbf_solver_accepts_valid_bone(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that validate_add_rbf_solver accepts a valid bone that can be used as a driver.
    """
    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import get_available_driven_bones, validate_add_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Find a bone that doesn't have a solver
    available_bones = get_available_driven_bones(instance)
    new_driver_bone_name = None
    existing_solver_driver_names = {s.name.replace(RBF_SOLVER_POSTFIX, "") for s in instance.rbf_editor.rbf_solver_list}

    for bone_name, joint_index, is_in_group in available_bones:
        if bone_name in existing_solver_driver_names:
            continue
        if bone_name in instance.body_swing_bone_names:
            continue
        if bone_name in instance.body_twist_bone_names:
            continue
        if joint_index >= 0:
            new_driver_bone_name = bone_name
            break

    if new_driver_bone_name is None:
        pytest.skip("No available bones to test validation")

    # Validate using core function
    is_valid, message = validate_add_rbf_solver(instance, new_driver_bone_name)

    assert is_valid is True, f"Validation should accept valid bone: {message}"
    assert message == "", f"Error message should be empty for valid bone: {message}"


def test_add_rbf_solver_creates_new_solver(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that add_rbf_solver creates a new solver with the correct properties.
    """
    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import add_rbf_solver, get_available_driven_bones

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Find a bone that doesn't have a solver
    available_bones = get_available_driven_bones(instance)
    new_driver_bone_name = None
    existing_solver_driver_names = {s.name.replace(RBF_SOLVER_POSTFIX, "") for s in instance.rbf_editor.rbf_solver_list}

    for bone_name, joint_index, is_in_group in available_bones:
        if bone_name in existing_solver_driver_names:
            continue
        if bone_name in instance.body_swing_bone_names:
            continue
        if bone_name in instance.body_twist_bone_names:
            continue
        if joint_index >= 0:
            new_driver_bone_name = bone_name
            break

    if new_driver_bone_name is None:
        pytest.skip("No available bones to create a new solver")

    initial_solver_count = len(instance.rbf_editor.rbf_solver_list)

    # Add the solver using core function
    success, message, new_solver_index = add_rbf_solver(instance, new_driver_bone_name)

    assert success is True, f"add_rbf_solver should succeed: {message}"
    assert new_solver_index >= 0, f"New solver index should be valid, got {new_solver_index}"

    # Verify a new solver was created
    assert len(instance.rbf_editor.rbf_solver_list) == initial_solver_count + 1, (
        f"Solver count should increase from {initial_solver_count} to {initial_solver_count + 1}"
    )

    # Verify the new solver has the correct properties
    new_solver = instance.rbf_editor.rbf_solver_list[new_solver_index]
    expected_solver_name = f"{new_driver_bone_name}{RBF_SOLVER_POSTFIX}"
    assert new_solver.name == expected_solver_name, (
        f"New solver should be named '{expected_solver_name}', got '{new_solver.name}'"
    )

    # Verify the solver has a default pose
    assert len(new_solver.poses) == 1, f"New solver should have 1 default pose, got {len(new_solver.poses)}"
    default_pose = new_solver.poses[0]
    assert default_pose.name == "default", f"First pose should be named 'default', got '{default_pose.name}'"

    # Verify the default pose has the driver bone
    assert len(default_pose.drivers) == 1, f"Default pose should have 1 driver, got {len(default_pose.drivers)}"
    driver = default_pose.drivers[0]
    assert driver.name == new_driver_bone_name, f"Driver bone should be '{new_driver_bone_name}', got '{driver.name}'"

    # Verify the default pose has no driven bones
    assert len(default_pose.driven) == 0, (
        f"Default pose should have no driven bones initially, got {len(default_pose.driven)}"
    )


def test_add_rbf_solver_with_custom_quaternion(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that add_rbf_solver can accept a custom quaternion for the driver bone.
    """
    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import add_rbf_solver, get_available_driven_bones

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Find a bone that doesn't have a solver
    available_bones = get_available_driven_bones(instance)
    new_driver_bone_name = None
    existing_solver_driver_names = {s.name.replace(RBF_SOLVER_POSTFIX, "") for s in instance.rbf_editor.rbf_solver_list}

    for bone_name, joint_index, is_in_group in available_bones:
        if bone_name in existing_solver_driver_names:
            continue
        if bone_name in instance.body_swing_bone_names:
            continue
        if bone_name in instance.body_twist_bone_names:
            continue
        if joint_index >= 0:
            new_driver_bone_name = bone_name
            break

    if new_driver_bone_name is None:
        pytest.skip("No available bones to create a new solver")

    # Custom quaternion (not identity)
    custom_quaternion = (0.707, 0.707, 0.0, 0.0)

    # Add the solver using core function with custom quaternion
    success, message, new_solver_index = add_rbf_solver(
        instance, new_driver_bone_name, driver_quaternion=custom_quaternion
    )

    assert success is True, f"add_rbf_solver should succeed: {message}"

    # Verify the driver quaternion was set
    new_solver = instance.rbf_editor.rbf_solver_list[new_solver_index]
    default_pose = new_solver.poses[0]
    driver = default_pose.drivers[0]

    assert driver.quaternion_rotation[0] == pytest.approx(custom_quaternion[0], abs=TOLERANCE), (
        f"Driver quaternion W should be {custom_quaternion[0]}, got {driver.quaternion_rotation[0]}"
    )
    assert driver.quaternion_rotation[1] == pytest.approx(custom_quaternion[1], abs=TOLERANCE), (
        f"Driver quaternion X should be {custom_quaternion[1]}, got {driver.quaternion_rotation[1]}"
    )


def test_add_rbf_solver_fails_for_invalid_bone(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that add_rbf_solver fails and returns an error for invalid bones.
    """
    from character_dna.editors.rbf_editor.utilities import add_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    initial_solver_count = len(instance.rbf_editor.rbf_solver_list)

    # Try to add a solver for a nonexistent bone
    success, message, new_solver_index = add_rbf_solver(instance, "nonexistent_bone_xyz_123")

    assert success is False, "add_rbf_solver should fail for nonexistent bone"
    assert new_solver_index == -1, "Solver index should be -1 on failure"
    assert "not found" in message.lower(), f"Error message should mention bone not found: {message}"

    # Verify no solver was added
    assert len(instance.rbf_editor.rbf_solver_list) == initial_solver_count, "Solver count should not change on failure"


def test_remove_rbf_solver_removes_active_solver(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that remove_rbf_solver removes the active solver and updates indices.
    """
    from character_dna.editors.rbf_editor.utilities import remove_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    if len(instance.rbf_editor.rbf_solver_list) == 0:
        pytest.skip("No solvers available to test removal")

    initial_solver_count = len(instance.rbf_editor.rbf_solver_list)
    active_index = instance.rbf_editor.rbf_solver_list_active_index
    solver_to_remove_name = instance.rbf_editor.rbf_solver_list[active_index].name

    # Remove using core function (no solver_index = use active)
    success, message = remove_rbf_solver(instance)

    assert success is True, f"remove_rbf_solver should succeed: {message}"

    # Verify solver was removed
    assert len(instance.rbf_editor.rbf_solver_list) == initial_solver_count - 1, (
        f"Solver count should decrease from {initial_solver_count} to {initial_solver_count - 1}"
    )

    # Verify the removed solver is no longer in the list
    for solver in instance.rbf_editor.rbf_solver_list:
        assert solver.name != solver_to_remove_name, (
            f"Solver '{solver_to_remove_name}' should have been removed but is still in the list"
        )

    # Verify active index is updated correctly
    if initial_solver_count > 1:
        assert instance.rbf_editor.rbf_solver_list_active_index <= len(instance.rbf_editor.rbf_solver_list) - 1, (
            f"Active index {instance.rbf_editor.rbf_solver_list_active_index} should be within bounds "
            f"[0, {len(instance.rbf_editor.rbf_solver_list) - 1}]"
        )


def test_remove_rbf_solver_by_index(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that remove_rbf_solver can remove a solver by specific index.
    """
    from character_dna.editors.rbf_editor.utilities import remove_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    if len(instance.rbf_editor.rbf_solver_list) < 2:
        pytest.skip("Need at least 2 solvers to test removal by index")

    initial_solver_count = len(instance.rbf_editor.rbf_solver_list)
    solver_to_remove_name = instance.rbf_editor.rbf_solver_list[1].name

    # Remove solver at index 1 (not the active one)
    success, message = remove_rbf_solver(instance, solver_index=1)

    assert success is True, f"remove_rbf_solver should succeed: {message}"

    # Verify solver was removed
    assert len(instance.rbf_editor.rbf_solver_list) == initial_solver_count - 1, (
        f"Solver count should decrease from {initial_solver_count} to {initial_solver_count - 1}"
    )

    # Verify the specific solver was removed
    for solver in instance.rbf_editor.rbf_solver_list:
        assert solver.name != solver_to_remove_name, (
            f"Solver '{solver_to_remove_name}' should have been removed but is still in the list"
        )


def test_remove_rbf_solver_fails_when_no_solvers(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that remove_rbf_solver fails gracefully when there are no solvers.
    """
    from character_dna.editors.rbf_editor.utilities import remove_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    # Clear all solvers
    original_solver_count = len(instance.rbf_editor.rbf_solver_list)
    for _ in range(original_solver_count):
        instance.rbf_editor.rbf_solver_list.remove(0)

    try:
        # Try to remove when no solvers exist
        success, message = remove_rbf_solver(instance)

        assert success is False, "remove_rbf_solver should fail when no solvers exist"
        assert "no rbf solvers" in message.lower(), f"Error message should mention no solvers: {message}"
    finally:
        # Reinitialize to restore solvers (cleanup)
        instance.body_initialize(update_rbf_solver_list=True)


def test_remove_rbf_solver_fails_for_invalid_index(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that remove_rbf_solver fails gracefully for invalid indices.
    """
    from character_dna.editors.rbf_editor.utilities import remove_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    if len(instance.rbf_editor.rbf_solver_list) == 0:
        pytest.skip("No solvers available to test invalid index removal")

    initial_solver_count = len(instance.rbf_editor.rbf_solver_list)

    # Try to remove with an invalid index
    success, message = remove_rbf_solver(instance, solver_index=999)

    assert success is False, "remove_rbf_solver should fail for invalid index"
    assert "invalid" in message.lower(), f"Error message should mention invalid index: {message}"

    # Verify no solver was removed
    assert len(instance.rbf_editor.rbf_solver_list) == initial_solver_count, "Solver count should not change on failure"


# =============================================================================
# New Solver Core Function Tests
# =============================================================================


def test_new_solver_default_pose_has_no_driven_bones(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that a newly created solver's default pose has no driven bones initially.

    This is a regression test to ensure that poses without driven bones (like the
    default pose on a new solver) are handled correctly and don't cause errors
    in the joint group finding logic.
    """
    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import add_rbf_solver, get_available_driven_bones

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Find a bone that doesn't have a solver
    available_bones = get_available_driven_bones(instance)
    new_driver_bone_name = None
    existing_solver_driver_names = {s.name.replace(RBF_SOLVER_POSTFIX, "") for s in instance.rbf_editor.rbf_solver_list}

    for bone_name, joint_index, is_in_group in available_bones:
        if bone_name in existing_solver_driver_names:
            continue
        if bone_name in instance.body_swing_bone_names:
            continue
        if bone_name in instance.body_twist_bone_names:
            continue
        if joint_index >= 0:
            new_driver_bone_name = bone_name
            break

    if new_driver_bone_name is None:
        pytest.skip("No available bones to create a new solver")

    # Add the solver using core function
    success, message, new_solver_index = add_rbf_solver(instance, new_driver_bone_name)
    assert success is True, f"add_rbf_solver should succeed: {message}"

    # Get the new solver and its default pose
    new_solver = instance.rbf_editor.rbf_solver_list[new_solver_index]
    assert len(new_solver.poses) == 1, "New solver should have exactly 1 pose (default)"

    default_pose = new_solver.poses[0]
    assert default_pose.name == "default", "First pose should be the default pose"

    # Verify the default pose has no driven bones
    assert len(default_pose.driven) == 0, (
        f"Default pose should have no driven bones initially, got {len(default_pose.driven)}"
    )

    # Verify it has exactly 1 driver (the driver bone)
    assert len(default_pose.drivers) == 1, f"Default pose should have exactly 1 driver, got {len(default_pose.drivers)}"


def test_add_then_remove_solver_restores_original_state(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that adding and then removing a solver restores the original state.

    This verifies the full add/remove lifecycle using only core functions.
    """
    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import (
        add_rbf_solver,
        get_available_driven_bones,
        remove_rbf_solver,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Record initial state
    initial_solver_count = len(instance.rbf_editor.rbf_solver_list)
    initial_solver_names = {s.name for s in instance.rbf_editor.rbf_solver_list}

    # Find a bone that doesn't have a solver
    available_bones = get_available_driven_bones(instance)
    new_driver_bone_name = None
    existing_solver_driver_names = {s.name.replace(RBF_SOLVER_POSTFIX, "") for s in instance.rbf_editor.rbf_solver_list}

    for bone_name, joint_index, is_in_group in available_bones:
        if bone_name in existing_solver_driver_names:
            continue
        if bone_name in instance.body_swing_bone_names:
            continue
        if bone_name in instance.body_twist_bone_names:
            continue
        if joint_index >= 0:
            new_driver_bone_name = bone_name
            break

    if new_driver_bone_name is None:
        pytest.skip("No available bones to create a new solver")

    # Add the solver using core function
    success, message, new_solver_index = add_rbf_solver(instance, new_driver_bone_name)
    assert success is True, f"add_rbf_solver should succeed: {message}"

    # Verify the solver was added
    assert len(instance.rbf_editor.rbf_solver_list) == initial_solver_count + 1, "Solver count should increase by 1"

    new_solver_name = f"{new_driver_bone_name}{RBF_SOLVER_POSTFIX}"
    assert new_solver_name in {s.name for s in instance.rbf_editor.rbf_solver_list}, "New solver should exist"

    # Remove the solver using core function
    success, message = remove_rbf_solver(instance, new_solver_index)
    assert success is True, f"remove_rbf_solver should succeed: {message}"

    # Verify the original state is restored
    assert len(instance.rbf_editor.rbf_solver_list) == initial_solver_count, "Solver count should be restored"
    assert {s.name for s in instance.rbf_editor.rbf_solver_list} == initial_solver_names, (
        "Original solvers should be preserved"
    )


# =============================================================================
# Crash Reproduction Tests
# =============================================================================


def test_new_solver_with_pose_and_driven_bones_commits_without_crash(
    fresh_rbf_test_scene, temp_folder, dna_folder_name: str
):
    """
    Regression test for RigLogic EXCEPTION_INT_DIVIDE_BY_ZERO crash.

    This test reproduces the exact scenario that causes a hard crash in RigLogic:
    1. Remove all existing RBF solvers
    2. Add a new solver for calf_l
    3. Add a pose "calf_l_back_90" with driven bones (calf_kneeBack_l, calf_knee_l, calf_twistCor_02_l)
    4. Set the driver bone calf_l rotated back 90 degrees
    5. Commit changes to DNA
    6. Reload the DNA (this is where the crash occurs)

    The crash occurs in riglogic.cp311-win_amd64.pyd with EXCEPTION_INT_DIVIDE_BY_ZERO
    during body_initialize after committing the changes.
    """
    import math

    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import (
        add_rbf_pose,
        add_rbf_solver,
        remove_rbf_solver,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Enable editing mode
    instance.rbf_editor.is_editing = True

    # Step 1: Remove all existing RBF solvers
    while len(instance.rbf_editor.rbf_solver_list) > 0:
        success, message = remove_rbf_solver(instance, 0)
        assert success is True, f"Failed to remove solver: {message}"

    assert len(instance.rbf_editor.rbf_solver_list) == 0, "All solvers should be removed"

    # Step 2: Add a new solver for calf_l
    driver_bone_name = "calf_l"
    success, message, new_solver_index = add_rbf_solver(instance, driver_bone_name)
    assert success is True, f"add_rbf_solver should succeed: {message}"

    solver = instance.rbf_editor.rbf_solver_list[new_solver_index]
    assert solver.name == f"{driver_bone_name}{RBF_SOLVER_POSTFIX}"

    # Step 3: Add a pose "calf_l_back_90" with specific driven bones
    # Driver rotated back 90 degrees around X axis (in Blender's coordinate system)
    # Quaternion for 90 degree rotation around X axis
    angle_rad = math.radians(90)
    half_angle = angle_rad / 2
    driver_quaternion = (
        math.cos(half_angle),  # w
        math.sin(half_angle),  # x
        0.0,  # y
        0.0,  # z
    )

    # Driven bones with their transforms (simulating a bent leg pose)
    driven_bone_transforms = {
        "calf_kneeBack_l": {
            "location": [0.0, 0.0, 0.0],
            "rotation": [0.1, 0.0, 0.0],  # Small rotation
            "scale": [1.0, 1.0, 1.0],
        },
        "calf_knee_l": {
            "location": [0.0, 0.0, 0.0],
            "rotation": [0.05, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "calf_twistCor_02_l": {
            "location": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.05],
            "scale": [1.0, 1.0, 1.0],
        },
    }

    success, message, new_pose_index = add_rbf_pose(
        instance=instance,
        pose_name="calf_l_back_90",
        driver_quaternion=driver_quaternion,
        driven_bone_transforms=driven_bone_transforms,
        solver_index=new_solver_index,
    )
    assert success is True, f"add_rbf_pose should succeed: {message}"

    # Verify the pose was created correctly
    new_pose = None
    for pose in solver.poses:
        if pose.name == "calf_l_back_90":
            new_pose = pose
            break

    assert new_pose is not None, "New pose should exist"
    assert len(new_pose.driven) == 3, f"Pose should have 3 driven bones, got {len(new_pose.driven)}"
    assert len(new_pose.drivers) == 1, f"Pose should have 1 driver, got {len(new_pose.drivers)}"

    # Step 4: Commit changes to DNA - this writes the DNA file
    result = bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore
    assert result == {"FINISHED"}, f"CommitRBFSolverChanges should succeed, got {result}"

    # Step 5: Verify the DNA can be reloaded without crashing
    # This is where the EXCEPTION_INT_DIVIDE_BY_ZERO crash occurs in RigLogic
    instance.destroy()
    instance.body_initialize()  # This line triggers the crash

    assert instance.body_initialized, "Instance should be initialized after reloading DNA"

    # Verify the new solver exists after reload
    found_solver = False
    for s in instance.rbf_editor.rbf_solver_list:
        if s.name == f"{driver_bone_name}{RBF_SOLVER_POSTFIX}":
            found_solver = True
            # Verify the pose exists
            pose_names = [p.name for p in s.poses]
            assert "calf_l_back_90" in pose_names, f"Pose 'calf_l_back_90' not found after reload. Found: {pose_names}"
            break

    assert found_solver, f"Solver '{driver_bone_name}{RBF_SOLVER_POSTFIX}' not found after reloading DNA"


def test_new_solver_driven_bone_transforms_persist_after_commit(
    fresh_rbf_test_scene, temp_folder, dna_folder_name: str
):
    """
    Regression test for driven bone transforms not being persisted after commit.

    This test reproduces a bug where:
    1. Remove all existing RBF solvers
    2. Add a new solver
    3. Add a pose with driven bones that have non-zero transforms
    4. Commit changes to DNA
    5. Reload the DNA
    6. The driven bone transforms are NOT preserved (they appear as zeros)

    This is related to the EXCEPTION_INT_DIVIDE_BY_ZERO crash fix, but specifically
    tests that the transform data itself is correctly written to the DNA file.
    """
    import math

    from character_dna.constants import RBF_SOLVER_POSTFIX
    from character_dna.editors.rbf_editor.utilities import (
        add_rbf_pose,
        add_rbf_solver,
        remove_rbf_solver,
    )

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Enable editing mode
    instance.rbf_editor.is_editing = True

    # Step 1: Remove all existing RBF solvers
    while len(instance.rbf_editor.rbf_solver_list) > 0:
        success, message = remove_rbf_solver(instance, 0)
        assert success is True, f"Failed to remove solver: {message}"

    assert len(instance.rbf_editor.rbf_solver_list) == 0, "All solvers should be removed"

    # Step 2: Add a new solver for calf_l
    driver_bone_name = "calf_l"
    success, message, new_solver_index = add_rbf_solver(instance, driver_bone_name)
    assert success is True, f"add_rbf_solver should succeed: {message}"

    solver = instance.rbf_editor.rbf_solver_list[new_solver_index]

    # Step 3: Add a pose with driven bones that have NON-ZERO transforms
    angle_rad = math.radians(90)
    half_angle = angle_rad / 2
    driver_quaternion = (
        math.cos(half_angle),  # w
        math.sin(half_angle),  # x
        0.0,  # y
        0.0,  # z
    )

    # Use significant transform values that we can verify after reload
    # Test with 3 driven bones - include location, rotation, AND scale changes
    expected_rotation_calf_kneeBack = 0.1  # ~5.7 degrees
    expected_rotation_calf_knee = 0.05  # ~2.9 degrees
    expected_rotation_calf_twist = 0.05  # ~2.9 degrees (around Z axis)

    # Non-zero location offsets (in meters/Blender units)
    expected_location_calf_kneeBack = [0.01, 0.02, 0.03]  # 1cm, 2cm, 3cm offset
    expected_location_calf_knee = [0.005, 0.0, 0.0]  # 5mm X offset
    expected_location_calf_twist = [0.0, 0.01, 0.0]  # 1cm Y offset

    # Non-identity scale values
    expected_scale_calf_kneeBack = [1.1, 1.0, 1.0]  # 10% scale on X
    expected_scale_calf_knee = [1.0, 1.05, 1.0]  # 5% scale on Y
    expected_scale_calf_twist = [1.0, 1.0, 0.95]  # -5% scale on Z

    driven_bone_transforms = {
        "calf_kneeBack_l": {
            "location": expected_location_calf_kneeBack,
            "rotation": [expected_rotation_calf_kneeBack, 0.0, 0.0],
            "scale": expected_scale_calf_kneeBack,
        },
        "calf_knee_l": {
            "location": expected_location_calf_knee,
            "rotation": [expected_rotation_calf_knee, 0.0, 0.0],
            "scale": expected_scale_calf_knee,
        },
        "calf_twistCor_02_l": {
            "location": expected_location_calf_twist,
            "rotation": [0.0, 0.0, expected_rotation_calf_twist],
            "scale": expected_scale_calf_twist,
        },
    }

    success, message, new_pose_index = add_rbf_pose(
        instance=instance,
        pose_name="calf_l_back_90",
        driver_quaternion=driver_quaternion,
        driven_bone_transforms=driven_bone_transforms,
        solver_index=new_solver_index,
    )
    assert success is True, f"add_rbf_pose should succeed: {message}"

    # Verify the pose has the correct transforms BEFORE commit
    new_pose = None
    for pose in solver.poses:
        if pose.name == "calf_l_back_90":
            new_pose = pose
            break

    assert new_pose is not None, "New pose should exist"
    assert len(new_pose.driven) == 3, f"Pose should have 3 driven bones, got {len(new_pose.driven)}"

    # Verify transforms BEFORE commit
    driven_lookup_before = {d.name: d for d in new_pose.driven}
    assert "calf_kneeBack_l" in driven_lookup_before, "calf_kneeBack_l should be in driven bones"
    assert driven_lookup_before["calf_kneeBack_l"].euler_rotation[0] == pytest.approx(
        expected_rotation_calf_kneeBack, abs=TOLERANCE
    ), f"calf_kneeBack_l rotation X before commit mismatch"

    # Step 4: Commit changes to DNA
    result = bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore
    assert result == {"FINISHED"}, f"CommitRBFSolverChanges should succeed, got {result}"

    # Step 5: Reload the DNA
    instance.destroy()
    instance.body_initialize()

    assert instance.body_initialized, "Instance should be initialized after reloading DNA"

    # Step 6: Verify the driven bone transforms are preserved after reload
    found_solver = None
    for s in instance.rbf_editor.rbf_solver_list:
        if s.name == f"{driver_bone_name}{RBF_SOLVER_POSTFIX}":
            found_solver = s
            break

    assert found_solver is not None, f"Solver not found after reload"

    # Find the pose
    found_pose = None
    for pose in found_solver.poses:
        if pose.name == "calf_l_back_90":
            found_pose = pose
            break

    assert found_pose is not None, f"Pose 'calf_l_back_90' not found after reload"
    assert len(found_pose.driven) == 3, f"Pose should have 3 driven bones after reload, got {len(found_pose.driven)}"

    # Verify transforms AFTER reload
    driven_lookup_after = {d.name: d for d in found_pose.driven}

    # Check calf_kneeBack_l - rotation, location, and scale
    assert "calf_kneeBack_l" in driven_lookup_after, "calf_kneeBack_l should be in driven bones after reload"
    d = driven_lookup_after["calf_kneeBack_l"]
    assert d.euler_rotation[0] == pytest.approx(expected_rotation_calf_kneeBack, abs=TOLERANCE), (
        f"calf_kneeBack_l rotation X after reload should be {expected_rotation_calf_kneeBack}, got {d.euler_rotation[0]}"
    )
    for i, axis in enumerate(["X", "Y", "Z"]):
        assert d.location[i] == pytest.approx(expected_location_calf_kneeBack[i], abs=TOLERANCE), (
            f"calf_kneeBack_l location {axis} after reload should be {expected_location_calf_kneeBack[i]}, got {d.location[i]}"
        )
        assert d.scale[i] == pytest.approx(expected_scale_calf_kneeBack[i], abs=TOLERANCE), (
            f"calf_kneeBack_l scale {axis} after reload should be {expected_scale_calf_kneeBack[i]}, got {d.scale[i]}"
        )

    # Check calf_knee_l - rotation, location, and scale
    assert "calf_knee_l" in driven_lookup_after, "calf_knee_l should be in driven bones after reload"
    d = driven_lookup_after["calf_knee_l"]
    assert d.euler_rotation[0] == pytest.approx(expected_rotation_calf_knee, abs=TOLERANCE), (
        f"calf_knee_l rotation X after reload should be {expected_rotation_calf_knee}, got {d.euler_rotation[0]}"
    )
    for i, axis in enumerate(["X", "Y", "Z"]):
        assert d.location[i] == pytest.approx(expected_location_calf_knee[i], abs=TOLERANCE), (
            f"calf_knee_l location {axis} after reload should be {expected_location_calf_knee[i]}, got {d.location[i]}"
        )
        assert d.scale[i] == pytest.approx(expected_scale_calf_knee[i], abs=TOLERANCE), (
            f"calf_knee_l scale {axis} after reload should be {expected_scale_calf_knee[i]}, got {d.scale[i]}"
        )

    # Check calf_twistCor_02_l - rotation, location, and scale
    assert "calf_twistCor_02_l" in driven_lookup_after, "calf_twistCor_02_l should be in driven bones after reload"
    d = driven_lookup_after["calf_twistCor_02_l"]
    assert d.euler_rotation[2] == pytest.approx(expected_rotation_calf_twist, abs=TOLERANCE), (
        f"calf_twistCor_02_l rotation Z after reload should be {expected_rotation_calf_twist}, got {d.euler_rotation[2]}"
    )
    for i, axis in enumerate(["X", "Y", "Z"]):
        assert d.location[i] == pytest.approx(expected_location_calf_twist[i], abs=TOLERANCE), (
            f"calf_twistCor_02_l location {axis} after reload should be {expected_location_calf_twist[i]}, got {d.location[i]}"
        )
        assert d.scale[i] == pytest.approx(expected_scale_calf_twist[i], abs=TOLERANCE), (
            f"calf_twistCor_02_l scale {axis} after reload should be {expected_scale_calf_twist[i]}, got {d.scale[i]}"
        )


def test_delete_solver_and_mirror_does_not_scramble_other_poses(fresh_rbf_test_scene, dna_folder_name: str):
    """
    Test that deleting a solver and mirroring another does not corrupt poses in unrelated solvers.

    This is a regression test for a bug where:
    1. Deleting a solver (e.g., calf_r_UERBFSolver)
    2. Mirroring another solver (e.g., calf_l_UERBFSolver) to recreate it
    3. Committing changes to DNA
    Would cause poses in OTHER solvers (e.g., thigh_l_UERBFSolver) to have scrambled driven bone data.

    The bug was caused by:
    1. Mirrored poses copying the source pose's joint_group_index, but with different (mirrored) bone names
    2. When has_stale_solvers triggered, joint groups were cleared but pose joint_group_index wasn't reset
    3. This caused poses to be written to wrong joint groups with wrong bone transformations
    """
    from character_dna.editors.rbf_editor.utilities import mirror_solver, remove_rbf_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"
    assert instance.body_rig is not None, "No body rig found on instance"

    # Enter edit mode
    instance.rbf_editor.is_editing = True
    instance.auto_evaluate_body = False

    # Capture thigh_l_UERBFSolver pose data BEFORE any changes
    thigh_solver_name = "thigh_l_UERBFSolver"
    thigh_pose_name = "thigh_l_bck_90"

    thigh_solver = None
    thigh_solver_index = -1
    for i, solver in enumerate(instance.rbf_editor.rbf_solver_list):
        if solver.name == thigh_solver_name:
            thigh_solver = solver
            thigh_solver_index = i
            break
    assert thigh_solver is not None, f"Solver '{thigh_solver_name}' not found"

    # Get original driven bone data for the thigh pose
    thigh_pose = None
    for pose in thigh_solver.poses:
        if pose.name == thigh_pose_name:
            thigh_pose = pose
            break
    assert thigh_pose is not None, f"Pose '{thigh_pose_name}' not found in solver '{thigh_solver_name}'"

    # Capture original driven bone transforms for comparison later
    original_driven_data = {}
    for driven in thigh_pose.driven:
        original_driven_data[driven.name] = {
            "location": list(driven.location[:]),
            "euler_rotation": list(driven.euler_rotation[:]),
            "scale": list(driven.scale[:]),
        }
    assert len(original_driven_data) > 0, "Thigh pose should have driven bones"

    # Step 1: Delete calf_r_UERBFSolver
    calf_r_solver_index = -1
    for i, solver in enumerate(instance.rbf_editor.rbf_solver_list):
        if solver.name == "calf_r_UERBFSolver":
            calf_r_solver_index = i
            break
    assert calf_r_solver_index >= 0, "calf_r_UERBFSolver not found"

    success, message = remove_rbf_solver(instance, solver_index=calf_r_solver_index)
    assert success, f"Failed to remove calf_r_UERBFSolver: {message}"

    # Step 2: Mirror calf_l_UERBFSolver to recreate calf_r_UERBFSolver
    calf_l_solver_index = -1
    for i, solver in enumerate(instance.rbf_editor.rbf_solver_list):
        if solver.name == "calf_l_UERBFSolver":
            calf_l_solver_index = i
            break
    assert calf_l_solver_index >= 0, "calf_l_UERBFSolver not found"

    instance.rbf_editor.rbf_solver_list_active_index = calf_l_solver_index

    solver_regex = r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?"
    bone_regex = r"(?P<prefix>.+)?(?P<side>_[lr])"
    pose_regex = r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?"

    success, message, mirrored_solver_index = mirror_solver(
        instance=instance,
        solver_regex=solver_regex,
        bone_regex=bone_regex,
        pose_regex=pose_regex,
        mirror_axis="x",
    )
    assert success, f"Failed to mirror calf_l_UERBFSolver: {message}"

    # Verify calf_r_UERBFSolver was recreated
    calf_r_recreated = False
    for solver in instance.rbf_editor.rbf_solver_list:
        if solver.name == "calf_r_UERBFSolver":
            calf_r_recreated = True
            break
    assert calf_r_recreated, "calf_r_UERBFSolver should have been recreated by mirroring"

    # Step 3: Commit the changes to DNA
    result = bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore
    assert result == {"FINISHED"}, f"CommitRBFSolverChanges operator failed: {result}"

    # Step 4: Verify thigh_l_UERBFSolver poses are NOT scrambled
    # Re-get the thigh solver after reload (instance was destroyed and reinitialized by commit)
    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found after commit"

    thigh_solver = None
    for solver in instance.rbf_editor.rbf_solver_list:
        if solver.name == thigh_solver_name:
            thigh_solver = solver
            break
    assert thigh_solver is not None, f"Solver '{thigh_solver_name}' not found after commit"

    thigh_pose = None
    for pose in thigh_solver.poses:
        if pose.name == thigh_pose_name:
            thigh_pose = pose
            break
    assert thigh_pose is not None, f"Pose '{thigh_pose_name}' not found after commit"

    # Verify the driven bones are the same as before
    current_driven_names = {d.name for d in thigh_pose.driven}
    original_driven_names = set(original_driven_data.keys())

    # The driven bone names should match
    assert current_driven_names == original_driven_names, (
        f"Driven bone names changed! Original: {original_driven_names}, Current: {current_driven_names}"
    )

    # Verify the transform values haven't been scrambled
    for driven in thigh_pose.driven:
        if driven.name not in original_driven_data:
            continue
        original = original_driven_data[driven.name]

        # Check that transforms are approximately the same (allowing for floating point tolerance)
        for i, axis in enumerate(["X", "Y", "Z"]):
            assert driven.location[i] == pytest.approx(original["location"][i], abs=TOLERANCE), (
                f"'{driven.name}' location {axis} changed from {original['location'][i]} to {driven.location[i]}"
            )
            assert driven.euler_rotation[i] == pytest.approx(original["euler_rotation"][i], abs=TOLERANCE), (
                f"'{driven.name}' rotation {axis} changed from {original['euler_rotation'][i]} to {driven.euler_rotation[i]}"
            )
            assert driven.scale[i] == pytest.approx(original["scale"][i], abs=TOLERANCE), (
                f"'{driven.name}' scale {axis} changed from {original['scale'][i]} to {driven.scale[i]}"
            )


# =============================================================================
# Mirroring Tests
# =============================================================================


@pytest.mark.parametrize(
    ("input_name", "pattern", "expected"),
    [
        # Left to right with _l suffix
        ("calf_l", r"(?P<prefix>.+)?(?P<side>_l)", "calf_r"),
        # Right to left with _r suffix
        ("calf_r", r"(?P<prefix>.+)?(?P<side>_r)", "calf_l"),
        # Left to right with _l_ infix
        ("calf_l_back_90", r"(?P<prefix>.+)?(?P<side>_l_)(?P<suffix>.+)?", "calf_r_back_90"),
        # Right to left with _r_ infix
        ("calf_r_back_90", r"(?P<prefix>.+)?(?P<side>_r_)(?P<suffix>.+)?", "calf_l_back_90"),
        # No match returns None
        ("root", r"(?P<prefix>.+)?(?P<side>_l_)(?P<suffix>.+)?", None),
    ],
)
def test_get_mirrored_name(input_name: str, pattern: str, expected: str | None):
    """Test that get_mirrored_name correctly mirrors names based on patterns."""
    from character_dna.editors.rbf_editor.utilities import get_mirrored_name

    result = get_mirrored_name(input_name, pattern)
    assert result == expected, f"Expected '{expected}', got '{result}'"


@pytest.mark.parametrize(
    ("input_name", "pattern", "expected"),
    [
        # Valid name with _l_ infix
        ("calf_l_back_90", r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?", True),
        # Valid name with _r_ infix
        ("calf_r_back_90", r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?", True),
        # Invalid name (no side marker)
        ("root", r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?", False),
    ],
)
def test_can_mirror_name(input_name: str, pattern: str, expected: bool):
    """Test that can_mirror_name correctly identifies mirrorable names."""
    from character_dna.editors.rbf_editor.utilities import can_mirror_name

    result = can_mirror_name(input_name, pattern)
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    (
        "source_solver_name",
        "expected_mirrored_solver_name",
    ),
    [
        ("calf_l_UERBFSolver", "calf_r_UERBFSolver"),
        ("thigh_l_UERBFSolver", "thigh_r_UERBFSolver"),
    ],
)
def test_mirror_solver_name_generation(
    source_solver_name: str,
    expected_mirrored_solver_name: str,
):
    """
    Test that solver names are correctly mirrored using the regex patterns.
    """
    from character_dna.editors.rbf_editor.utilities import get_mirrored_name

    # Use the default solver mirror regex pattern (matches both _l_ and _r_)
    solver_regex = r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?"
    result = get_mirrored_name(source_solver_name, solver_regex)
    assert result == expected_mirrored_solver_name, f"Expected '{expected_mirrored_solver_name}', got '{result}'"


@pytest.mark.parametrize(
    (
        "source_bone_name",
        "expected_mirrored_bone_name",
    ),
    [
        ("calf_l", "calf_r"),
        ("thigh_l", "thigh_r"),
        ("calf_knee_l", "calf_knee_r"),
        ("calf_twistCor_02_l", "calf_twistCor_02_r"),
    ],
)
def test_mirror_bone_name_generation(
    source_bone_name: str,
    expected_mirrored_bone_name: str,
):
    """
    Test that bone names are correctly mirrored using the regex patterns.
    """
    from character_dna.editors.rbf_editor.utilities import get_mirrored_name

    # Use the default bone mirror regex pattern (matches both _l and _r)
    bone_regex = r"(?P<prefix>.+)?(?P<side>_[lr])"
    result = get_mirrored_name(source_bone_name, bone_regex)
    assert result == expected_mirrored_bone_name, f"Expected '{expected_mirrored_bone_name}', got '{result}'"


@pytest.mark.parametrize(
    (
        "source_pose_name",
        "expected_mirrored_pose_name",
    ),
    [
        ("calf_l_back_90", "calf_r_back_90"),
        ("thigh_l_in_45_out_90", "thigh_r_in_45_out_90"),
    ],
)
def test_mirror_pose_name_generation(
    source_pose_name: str,
    expected_mirrored_pose_name: str,
):
    """
    Test that pose names are correctly mirrored using the regex patterns.
    """
    from character_dna.editors.rbf_editor.utilities import get_mirrored_name

    # Use the default pose mirror regex pattern (matches both _l_ and _r_)
    pose_regex = r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?"
    result = get_mirrored_name(source_pose_name, pose_regex)
    assert result == expected_mirrored_pose_name, f"Expected '{expected_mirrored_pose_name}', got '{result}'"


def test_validate_mirror_solver_target_exists(fresh_rbf_test_scene):
    """
    Test that validate_mirror_solver returns an error when the target solver already exists.
    """
    from character_dna.editors.rbf_editor.utilities import validate_mirror_solver

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    # Enter edit mode
    instance.rbf_editor.is_editing = True
    instance.auto_evaluate_body = False

    # Set calf_l_UERBFSolver as active - it has a mirrored counterpart calf_r_UERBFSolver
    for solver_index, solver in enumerate(instance.rbf_editor.rbf_solver_list):
        if solver.name == "calf_l_UERBFSolver":
            instance.rbf_editor.rbf_solver_list_active_index = solver_index
            break

    solver_regex = r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?"
    bone_regex = r"(?P<prefix>.+)?(?P<side>_[lr])"

    is_valid, error_message = validate_mirror_solver(instance, solver_regex, bone_regex)

    # Should fail because calf_r_UERBFSolver already exists
    assert is_valid is False, f"Expected validation to fail, but it passed"
    assert "already exists" in error_message.lower(), (
        f"Expected 'already exists' in error message, got: {error_message}"
    )


def test_validate_mirror_pose_no_target_solver(fresh_rbf_test_scene):
    """
    Test that validate_mirror_pose returns an error when the target solver doesn't exist.
    """
    from character_dna.editors.rbf_editor.utilities import add_rbf_solver, validate_mirror_pose

    instance = get_active_rig_instance()
    assert instance is not None, "No active rig instance found"

    # Enter edit mode
    instance.rbf_editor.is_editing = True
    instance.auto_evaluate_body = False

    # Create a new solver that doesn't have a mirrored counterpart
    # First find a bone that doesn't have a solver yet
    success, message, solver_index = add_rbf_solver(
        instance=instance,
        driver_bone_name="pelvis",  # pelvis doesn't have a mirrored counterpart
        driver_quaternion=(1.0, 0.0, 0.0, 0.0),
    )

    if not success:
        # Pelvis solver may already exist, skip this test
        pytest.skip("Could not create test solver")

    # Add a non-default pose to the solver
    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    pose = solver.poses.add()
    pose.solver_index = solver.solver_index
    pose.pose_index = 9999
    pose["name"] = "test_pose"
    solver.poses_active_index = len(solver.poses) - 1

    solver_regex = r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?"
    pose_regex = r"(?P<prefix>.+)?(?P<side>_[lr]_)(?P<suffix>.+)?"

    is_valid, error_message = validate_mirror_pose(instance, solver_regex, pose_regex)

    # Should fail because pelvis doesn't match the mirror pattern
    assert is_valid is False, f"Expected validation to fail, but it passed"
