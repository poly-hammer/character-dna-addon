from typing import Any, cast

import bpy
import pytest

from character_dna.constants import IS_BLENDER_5
from character_dna.ui.callbacks import get_active_rig_instance
from character_dna.utilities.action import FACE_BOARD_EXCLUDED_CONTROLS, get_channel_bag
from constants import TEST_ANIMATION_FOLDER


@pytest.mark.parametrize(
    ("component", "file_name"),
    [
        ("body", "MHC_BodyROM.fbx"),
        ("head", "MHC_HeadROM.fbx"),
    ],
)
def test_import_component_animation(load_full_dna_for_animation, component: str, file_name: str):
    instance = get_active_rig_instance()
    file_path = TEST_ANIMATION_FOLDER / component / file_name

    bpy.ops.character_dna.import_component_animation(component_type=component, filepath=str(file_path))

    if component == "body":
        assert instance.body_rig.animation_data.action.name == f"{instance.name}_{component}_{file_path.stem}"
    elif component == "head":
        assert instance.head_rig.animation_data.action.name == f"{instance.name}_{component}_{file_path.stem}"


@pytest.mark.parametrize(
    ("file_name"),
    [
        ("MHC_FaceBoardROM.fbx"),
    ],
)
def test_import_face_board_animation(load_full_dna_for_animation, file_name: str, caplog):
    instance = get_active_rig_instance()
    assert instance is not None
    file_path = TEST_ANIMATION_FOLDER / "head" / file_name

    bpy.ops.character_dna.import_face_board_animation(filepath=str(file_path))

    assert instance.face_board.animation_data.action.name == f"{instance.name}_face_board_{file_path.stem}"
    assert not any("no matching face board bone" in record.getMessage() for record in caplog.records)
    action = instance.face_board.animation_data.action
    curves = cast("Any", get_channel_bag(action, instance.face_board.name)).fcurves
    assert len(curves) > 0
    for curve in curves:
        name = curve.data_path.split('"', 2)[1]
        assert name.startswith("CTRL_")
        assert not name.endswith("_grp")
        assert name not in FACE_BOARD_EXCLUDED_CONTROLS


def test_body_animation_onto_face_board_is_rejected(load_full_dna_for_animation):
    instance = get_active_rig_instance()
    file_path = TEST_ANIMATION_FOLDER / "body" / "MHC_BodyROM.fbx"
    action_before = instance.face_board.animation_data.action if instance.face_board.animation_data else None

    # Headless runs have nobody to answer the confirmation dialog, so the import
    # refuses instead of guessing.
    with pytest.raises(RuntimeError, match="face board"):
        bpy.ops.character_dna.import_face_board_animation(filepath=str(file_path))

    action_after = instance.face_board.animation_data.action if instance.face_board.animation_data else None
    assert action_after == action_before


def test_face_board_animation_onto_body_is_rejected(load_full_dna_for_animation):
    instance = get_active_rig_instance()
    file_path = TEST_ANIMATION_FOLDER / "head" / "MHC_FaceBoardROM.fbx"
    action_before = instance.body_rig.animation_data.action if instance.body_rig.animation_data else None

    with pytest.raises(RuntimeError, match="body"):
        bpy.ops.character_dna.import_component_animation(component_type="body", filepath=str(file_path))

    action_after = instance.body_rig.animation_data.action if instance.body_rig.animation_data else None
    assert action_after == action_before


def test_validation_can_be_overridden(load_full_dna_for_animation):
    instance = get_active_rig_instance()
    file_path = TEST_ANIMATION_FOLDER / "body" / "MHC_BodyROM.fbx"

    result = bpy.ops.character_dna.import_face_board_animation(filepath=str(file_path), ignore_validation=True)

    assert result == {"FINISHED"}
    assert instance.face_board.animation_data.action.name == f"{instance.name}_face_board_{file_path.stem}"


@pytest.mark.parametrize(
    ("component", "action_name", "prefix_instance_name", "prefix_component_name", "replace_action"),
    [
        ("body", "test", True, True, False),
        ("body", "test", True, True, True),
        # ('head', 'test', True, True, False),
    ],
)
def test_bake_component_animation(
    load_full_dna_for_animation,
    component: str,
    action_name: str,
    prefix_instance_name: bool,
    prefix_component_name: bool,
    replace_action: bool,
):
    instance = get_active_rig_instance()
    bpy.context.window_manager.character_dna.current_component_type = component

    if IS_BLENDER_5:
        previous_object_action_names = [a.name for a in bpy.data.actions if a.slots[0].target_id_type == "OBJECT"]
        [a.name for a in bpy.data.actions if a.slots[0].target_id_type == "NODETREE"]
    else:
        previous_object_action_names = [a.name for a in bpy.data.actions if a.id_root == "OBJECT"]
        [a.name for a in bpy.data.actions if a.id_root == "NODETREE"]

    bpy.ops.character_dna.bake_component_animation(
        start_frame=1,
        end_frame=10,
        component_type=component,
        action_name=action_name,
        prefix_instance_name=prefix_instance_name,
        prefix_component_name=prefix_component_name,
        replace_action=replace_action,
    )

    if IS_BLENDER_5:
        expected_object_action_names = [a.name for a in bpy.data.actions if a.slots[0].target_id_type == "OBJECT"]
        [a.name for a in bpy.data.actions if a.slots[0].target_id_type == "NODETREE"]
    else:
        expected_object_action_names = [a.name for a in bpy.data.actions if a.id_root == "OBJECT"]
        [a.name for a in bpy.data.actions if a.id_root == "NODETREE"]

    new_object_actions = set(expected_object_action_names) - set(previous_object_action_names)

    if not replace_action:
        assert len(new_object_actions) == 1, "A new action should be created when not replacing an existing action."
        assert new_object_actions.pop() == f"{instance.name}_{component}_{action_name}", (
            "The baked action name is not as expected."
        )


@pytest.mark.parametrize(
    ("action_name", "prefix_instance_name", "prefix_component_name", "replace_action"),
    [
        ("face_board_test", True, True, False),
        ("face_board_test", True, True, True),
    ],
)
def test_bake_face_board_animation(
    load_full_dna_for_animation,
    action_name: str,
    prefix_instance_name: bool,
    prefix_component_name: bool,
    replace_action: bool,
):
    instance = get_active_rig_instance()

    if IS_BLENDER_5:
        previous_object_action_names = [
            a.name
            for a in bpy.data.actions
            if a.slots[0].target_id_type == "OBJECT" and a.name != f"{instance.name}_head_{action_name}"
        ]
        previous_node_tree_action_names = [
            a.name
            for a in bpy.data.actions
            if a.slots[0].target_id_type == "NODETREE"
            and a.name != f"{instance.name}_head_{action_name}_texture_logic_node"
        ]
    else:
        previous_object_action_names = [
            a.name
            for a in bpy.data.actions
            if a.id_root == "OBJECT" and a.name != f"{instance.name}_head_{action_name}"
        ]
        previous_node_tree_action_names = [
            a.name
            for a in bpy.data.actions
            if a.id_root == "NODETREE" and a.name != f"{instance.name}_head_{action_name}_texture_logic_node"
        ]

    bpy.ops.character_dna.bake_face_board_animation(
        start_frame=1,
        end_frame=10,
        action_name=action_name,
        prefix_instance_name=prefix_instance_name,
        prefix_component_name=prefix_component_name,
        replace_action=replace_action,
    )

    if IS_BLENDER_5:
        expected_object_action_names = [a.name for a in bpy.data.actions if a.slots[0].target_id_type == "OBJECT"]
        expected_node_tree_action_names = [a.name for a in bpy.data.actions if a.slots[0].target_id_type == "NODETREE"]
    else:
        expected_object_action_names = [a.name for a in bpy.data.actions if a.id_root == "OBJECT"]
        expected_node_tree_action_names = [a.name for a in bpy.data.actions if a.id_root == "NODETREE"]

    new_object_actions = set(expected_object_action_names) - set(previous_object_action_names)
    new_node_tree_action_names = set(expected_node_tree_action_names) - set(previous_node_tree_action_names)

    if not replace_action:
        assert len(new_object_actions) == 1, "A new action should be created when not replacing an existing action."

        assert new_object_actions.pop() == f"{instance.name}_head_{action_name}", (
            "The baked action name is not as expected."
        )

    assert len(new_node_tree_action_names) == 1, (
        "A new node tree action should always be created for face board baking."
    )
    assert any(
        name == f"{instance.name}_head_{action_name}_texture_logic_node" for name in expected_node_tree_action_names
    ), "The baked node tree action name is not as expected."


def test_bake_face_board_animation_ignores_unrelated_channels(load_full_dna_for_animation):
    instance = get_active_rig_instance()
    action = instance.face_board.animation_data.action
    channel_bag = get_channel_bag(action, action.name)
    # The face board can carry channels that are not pose bone transforms at all.
    channel_bag.fcurves.new(data_path="hide_viewport")

    result = bpy.ops.character_dna.bake_face_board_animation(
        start_frame=1,
        end_frame=3,
        action_name="stray_channel_test",
        prefix_instance_name=True,
        prefix_component_name=True,
        replace_action=False,
    )

    assert result == {"FINISHED"}
