# standard library imports

# third party imports
import pytest

# local imports
from character_dna.rig_definition import HeadRigDefinition


@pytest.fixture
def head_definition() -> HeadRigDefinition:
    return HeadRigDefinition.load()


def test_joint_groups_are_named_and_non_empty(head_definition: HeadRigDefinition):
    joint_groups = head_definition.joint_groups
    assert joint_groups, "Expected the head rig definition to carry joint groups."
    assert len(joint_groups) == head_definition.joint_group_count
    for joint_group in joint_groups:
        assert joint_group.name
        assert joint_group.joints, f"Joint group '{joint_group.name}' has no member joints."


def test_joint_group_members_exist_as_joints(head_definition: HeadRigDefinition):
    joint_names = set(head_definition.joint_names)
    for joint_group in head_definition.joint_groups:
        for joint_name in joint_group.joints:
            assert joint_name in joint_names


def test_color_by_joint_name_covers_colored_joints(head_definition: HeadRigDefinition):
    color_by_joint_name = head_definition.color_by_joint_name
    assert color_by_joint_name, "Expected the head rig definition joints to carry colors."

    colored_joints = {joint.name for joint in head_definition.joints if joint.color is not None}
    assert set(color_by_joint_name) == colored_joints

    for color in color_by_joint_name.values():
        assert len(color) >= 3
        assert all(0.0 <= channel <= 1.0 for channel in color)


def test_regions_are_named_and_reference_joints(head_definition: HeadRigDefinition):
    regions = head_definition.regions
    assert regions, "Expected the head rig definition to carry regions."
    assert len(regions) == len(head_definition.region_names)

    joint_names = set(head_definition.joint_names)
    for region in regions:
        assert region.name
        assert region.joints, f"Region '{region.name}' has no member joints."
        for joint_name in region.joints:
            assert joint_name in joint_names
