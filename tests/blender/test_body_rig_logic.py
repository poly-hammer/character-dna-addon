import pytest

from constants import EXCLUDE_FINGER_POSES
from utilities.rbf_editor import assert_body_pose, get_all_body_pose_names


@pytest.mark.parametrize(
    ("solver_name", "pose_name", "source_rig_name"),
    [
        (solver_name, pose_name, "ada_body_rig")
        for solver_name, pose_name in get_all_body_pose_names(exclude_fingers=EXCLUDE_FINGER_POSES)
    ],
)
def test_body_pose(
    load_body_dna,
    solver_name: str,
    pose_name: str,
    source_rig_name: str,
    show: bool = False,
    skip_fbx_import: bool = False,
):
    assert_body_pose(
        solver_name=solver_name,
        pose_name=pose_name,
        source_rig_name=source_rig_name,
        evaluate=True,
        show=show,
        skip_fbx_import=skip_fbx_import,
    )


@pytest.mark.parametrize(
    ("solver_name", "pose_name", "source_rig_name"),
    [
        (solver_name, pose_name, "ada_body_rig")
        for solver_name, pose_name in get_all_body_pose_names(exclude_fingers=EXCLUDE_FINGER_POSES)
    ],
)
def test_body_pose_edit_mode(
    load_body_dna,
    solver_name: str,
    pose_name: str,
    source_rig_name: str,
    show: bool = False,
    skip_fbx_import: bool = False,
):
    assert_body_pose(
        solver_name=solver_name,
        pose_name=pose_name,
        source_rig_name=source_rig_name,
        evaluate=False,
        show=show,
        skip_fbx_import=skip_fbx_import,
    )
