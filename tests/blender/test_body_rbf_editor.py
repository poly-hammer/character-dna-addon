import bpy
import pytest

from mathutils import Quaternion, Vector

from character_dna.ui.callbacks import get_active_rig_instance
from character_dna.utilities import reset_pose
from constants import EXCLUDE_FINGER_POSES
from utilities.rbf_editor import assert_body_pose, get_all_body_pose_names, set_body_pose


TOLERANCE = 1e-5


@pytest.mark.parametrize(
    ("solver_name", "pose_name", "source_rig_name"),
    [
        (solver_name, pose_name, "ada_body_rig")
        for solver_name, pose_name in get_all_body_pose_names(exclude_fingers=EXCLUDE_FINGER_POSES)
    ],
)
def test_body_pose_roundtrip(
    load_body_dna_for_pose_roundtrip,
    solver_name: str,
    pose_name: str,
    source_rig_name: str,
    show: bool = False,
    skip_fbx_import: bool = False,
):
    instance = get_active_rig_instance()

    # reset the pose to the default position
    reset_pose(instance.body_rig)

    _, solver_index, pose_index = set_body_pose(solver_name=solver_name, pose_name=pose_name)

    # update the rbf pose with the unmodified data
    bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore

    # commit these changes to the dna
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # reset the pose to the default position
    reset_pose(instance.body_rig)

    # now check if the pose still matches the original
    assert_body_pose(
        solver_name=solver_name,
        pose_name=pose_name,
        source_rig_name=source_rig_name,
        evaluate=True,
        show=show,
        skip_fbx_import=skip_fbx_import,
    )


@pytest.mark.parametrize(
    (
        "solver_name",
        "pose_name",
        "driver_bone_name",
        "driver_bone_rotation",
        "changed_driven_bone_names",
        "changed_driven_bone_locations",
    ),
    [
        (
            "calf_l_UERBFSolver",
            "calf_l_back_90",
            "calf_l",
            Quaternion((0.707107, 0.0, 0.0, -0.707107)),
            ["calf_twistCor_02_l"],
            [Vector((0.0, 0.1, 0.0))],
        ),
        (
            "thigh_l_UERBFSolver",
            "thigh_l_in_45_out_90",
            "thigh_l",
            Quaternion((0.653282, -0.270598, 0.270598, 0.653282)),
            ["thigh_out_l"],
            [Vector((0.0, 0.2, 0.0))],
        ),
        (
            "clavicle_r_UERBFSolver",
            "clavicle_r_up_40",
            "clavicle_r",
            Quaternion((0.939693, 0.0, -0.34202, 0.0)),
            ["clavicle_out_r"],
            [Vector((0.0, 0.1, 0.0))],
        ),
    ],
)
def test_body_pose_update(
    load_body_dna_for_pose_editing,
    solver_name: str,
    pose_name: str,
    driver_bone_name: str,
    driver_bone_rotation: Quaternion,
    changed_driven_bone_names: list[str],
    changed_driven_bone_locations: list[Vector],
):
    instance = get_active_rig_instance()

    # reset the pose to the default position
    reset_pose(instance.body_rig)

    pose, solver_index, pose_index = set_body_pose(solver_name=solver_name, pose_name=pose_name)

    for driven_name, change_location in zip(changed_driven_bone_names, changed_driven_bone_locations, strict=False):
        for driven_index, driven in enumerate(pose.driven):  # type: ignore
            if driven.name == driven_name:
                pose.driven_active_index = driven_index
                pose_bone = instance.body_rig.pose.bones[driven.name]

                # update the location
                pose_bone.location = change_location
                # update the driven bone transform in the pose
                bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore

    # commit these changes to the dna
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # reset the pose to the default position
    reset_pose(instance.body_rig)

    # set the driver bone rotation to the pose to trigger the driven bones
    driver_bone = instance.body_rig.pose.bones[driver_bone_name]
    # These historical driver fixtures are authored in Maya-local axes.
    w, x, y, z = driver_bone_rotation
    driver_bone.rotation_quaternion = Quaternion((w, x, -z, y))

    # ensure we evaluate the rig to apply the driven bone transforms
    instance.evaluate(component="body")

    for driven_name, expected_location in zip(changed_driven_bone_names, changed_driven_bone_locations, strict=False):
        pose_bone = instance.body_rig.pose.bones[driven_name]

        assert pose_bone.location.x == pytest.approx(expected_location.x, abs=TOLERANCE), (
            f"Driven bone {driven_name} X location {pose_bone.location.x} not {expected_location.x} as expected"
        )
        assert pose_bone.location.y == pytest.approx(expected_location.y, abs=TOLERANCE), (
            f"Driven bone {driven_name} Y location {pose_bone.location.y} not {expected_location.y} as expected"
        )
        assert pose_bone.location.z == pytest.approx(expected_location.z, abs=TOLERANCE), (
            f"Driven bone {driven_name} Z location {pose_bone.location.z} not {expected_location.z} as expected"
        )


@pytest.mark.parametrize(
    (
        "solver_name",
        "from_pose_name",
        "pose_name",
        "driver_bone_name",
        "driver_bone_rotation",
        "driven_bone_names",
        "changed_driven_bone_names",
        "changed_driven_bone_locations",
    ),
    [
        # (
        #     'calf_l_UERBFSolver',
        #     'calf_l_back_50',
        #     'calf_l_back_30',
        #     'calf_l',
        #     Quaternion((0.965926, 0.0, 0.0, -0.258819)),
        #     ['calf_knee_l', 'calf_kneeBack_l', 'calf_twistCor_02_l', 'thigh_twistCor_01_l', 'thigh_twistCor_02_l'],
        #     ['calf_twistCor_02_l'],
        #     [Vector((0.0, 0.1, 0.0))]
        # ),
    ],
)
def test_body_pose_duplicate(
    load_body_dna_for_pose_editing,
    solver_name: str,
    from_pose_name: str,
    pose_name: str,
    driver_bone_name: str,
    driver_bone_rotation: Quaternion,
    driven_bone_names: list[str],
    changed_driven_bone_names: list[str],
    changed_driven_bone_locations: list[Vector],
):
    instance = get_active_rig_instance()

    # reset the pose to the default position
    reset_pose(instance.body_rig)

    _, solver_index, from_pose_index = set_body_pose(solver_name=solver_name, pose_name=from_pose_name)

    # duplicate the pose to create a new one to edit
    bpy.ops.character_dna.duplicate_rbf_pose(solver_index=solver_index, pose_index=from_pose_index)

    solver = instance.rbf_editor.rbf_solver_list[solver_index]
    new_pose_index = len(solver.poses) - 1
    new_pose = solver.poses[new_pose_index]

    # rename the new pose after duplication
    new_pose.name = pose_name

    # set the driver bone rotation to the new value
    driver_bone = instance.body_rig.pose.bones[driver_bone_name]
    # These historical driver fixtures are authored in Maya-local axes.
    w, x, y, z = driver_bone_rotation
    driver_bone.rotation_quaternion = Quaternion((w, x, -z, y))

    for driven_name, change_location in zip(changed_driven_bone_names, changed_driven_bone_locations, strict=False):
        for driven_index, driven in enumerate(new_pose.driven):  # type: ignore
            if driven.name == driven_name:
                new_pose.driven_active_index = driven_index
                pose_bone = instance.body_rig.pose.bones[driven.name]

                # update the location
                pose_bone.location = change_location
                # update the driven bone transform in the pose
                bpy.ops.character_dna.apply_rbf_pose_edits()  # type: ignore

    # commit these changes to the dna
    bpy.ops.character_dna.commit_rbf_solver_changes()  # type: ignore

    # check that all expected driven bones are present in the duplicated pose
    new_pose, _, _ = set_body_pose(solver_name=solver_name, pose_name=pose_name)
    for driven in new_pose.driven:  # type: ignore
        driven_name = str(driven.name)
        assert driven_name in driven_bone_names, (
            f'Driven bone "{driven_name}" not in expected driven bone names after duplication'
        )

    # reset the pose to the default position
    reset_pose(instance.body_rig)

    # set the driver bone rotation to the pose to trigger the driven bones
    driver_bone = instance.body_rig.pose.bones[driver_bone_name]
    # These historical driver fixtures are authored in Maya-local axes.
    w, x, y, z = driver_bone_rotation
    driver_bone.rotation_quaternion = Quaternion((w, x, -z, y))

    # ensure we evaluate the rig to apply the driven bone transforms
    instance.evaluate(component="body")

    for driven_name, expected_location in zip(changed_driven_bone_names, changed_driven_bone_locations, strict=False):
        pose_bone = instance.body_rig.pose.bones[driven_name]

        assert pose_bone.location.x == pytest.approx(expected_location.x, abs=TOLERANCE), (
            f"Driven bone {driven_name} X location {pose_bone.location.x} not {expected_location.x} as expected"
        )
        assert pose_bone.location.y == pytest.approx(expected_location.y, abs=TOLERANCE), (
            f"Driven bone {driven_name} Y location {pose_bone.location.y} not {expected_location.y} as expected"
        )
        assert pose_bone.location.z == pytest.approx(expected_location.z, abs=TOLERANCE), (
            f"Driven bone {driven_name} Z location {pose_bone.location.z} not {expected_location.z} as expected"
        )
