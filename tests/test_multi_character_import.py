"""Importing a second character into a scene that already has an animated one.

The second character's face board is duplicated from the first, so it arrives carrying that
character's animation, expression and head constraint. All three have to be cleared before the
board is measured, or it gets placed relative to where it sat on the character it was copied
from rather than beside its own head.
"""

from pathlib import Path

import bpy
import pytest

from mathutils import Quaternion

from character_dna.runtime import engine
from character_dna.runtime.bindings import owned_curve
from character_dna.utilities.mesh import get_bounding_box_center, get_bounding_box_left_x, get_bounding_box_right_x
from constants import TEST_DNA_FOLDER


def import_dna(file_path: Path):
    lods = {f"import_lod{index}": index == 0 for index in range(8)}
    bpy.ops.character_dna.import_dna(  # type: ignore[attr-defined]
        filepath=str(file_path),
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_vertex_groups=True,
        import_materials=False,
        import_face_board=True,
        include_body=True,
        **lods,
    )


def animate_character(instance):
    """Move, bend and key the character, so its face board is far from where it was imported."""
    body_rig = instance.body_rig
    bpy.context.view_layer.objects.active = body_rig  # type: ignore[union-attr]
    bpy.ops.object.mode_set(mode="POSE")

    for bone_name, quaternion in (
        ("spine_04", Quaternion((0.5, 0.866, 0.0, 0.0))),
        ("neck_01", Quaternion((0.7071, 0.0, 0.7071, 0.0))),
        ("head", Quaternion((0.6, 0.4, 0.5, 0.47))),
    ):
        pose_bone = body_rig.pose.bones.get(bone_name)
        if pose_bone:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = quaternion
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=1)

    body_rig.location = (3.0, 2.0, 0.5)
    body_rig.keyframe_insert(data_path="location", frame=1)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = instance.face_board  # type: ignore[union-attr]
    bpy.ops.object.mode_set(mode="POSE")
    for control_name, value in (("CTRL_C_jaw", 0.8), ("CTRL_L_mouth_cornerPull", 1.0), ("CTRL_C_eye", 0.5)):
        pose_bone = instance.face_board.pose.bones.get(control_name)
        if pose_bone:
            pose_bone.location.y = value
            pose_bone.keyframe_insert(data_path="location", frame=1)

    # Blender 4.5 stores native eye visibility drivers on the armature data.
    # Exercise copied data animation on newer Blender versions as well.
    data = instance.face_board.data
    data["authored_signal"] = 0.5
    data.keyframe_insert(data_path='["authored_signal"]', frame=1)
    data["driven_signal"] = 0.0
    data.driver_add('["driven_signal"]').driver.expression = "0.5"

    bpy.context.scene.frame_set(1)  # type: ignore[union-attr]
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()  # type: ignore[union-attr]


@pytest.fixture(scope="session")
def animated_scene_with_a_second_character(addon):
    bpy.ops.wm.read_homefile(app_template="")
    for scene_object in list(bpy.data.objects):
        bpy.data.objects.remove(scene_object, do_unlink=True)

    import_dna(TEST_DNA_FOLDER / "ada" / "head.dna")
    scene_properties = bpy.context.scene.character_dna  # type: ignore[attr-defined]
    animate_character(scene_properties.rig_instance_list[0])

    import_dna(TEST_DNA_FOLDER / "default" / "head.dna")
    return scene_properties.rig_instance_list


def test_second_import_places_the_face_board_beside_its_own_head(animated_scene_with_a_second_character):
    instance = animated_scene_with_a_second_character[1]
    head_mesh = instance.head_mesh
    face_board = instance.face_board

    horizontal_gap = get_bounding_box_left_x(face_board) - get_bounding_box_right_x(head_mesh)
    vertical_offset = get_bounding_box_center(face_board).z - get_bounding_box_center(head_mesh).z

    assert horizontal_gap == pytest.approx(0.0, abs=1e-3), "face board is not flush with the right of the head"
    assert vertical_offset == pytest.approx(0.0, abs=1e-3), "face board is not centered on the head"


def test_duplicated_face_board_does_not_inherit_the_source_animation(animated_scene_with_a_second_character):
    source_instance, new_instance = animated_scene_with_a_second_character[:2]

    assert source_instance.face_board.animation_data.action is not None
    assert source_instance.face_board.data.animation_data.action is not None
    assert source_instance.face_board.data.animation_data.drivers.find('["driven_signal"]') is not None
    own_carriers = engine.carriers(new_instance)
    source_carriers = engine.carriers(source_instance)
    assert own_carriers
    for owner in (new_instance.face_board, new_instance.face_board.data):
        animation = owner.animation_data
        if animation is None:
            continue
        assert animation.action is None
        assert not animation.nla_tracks
        for curve in animation.drivers:
            assert any(owned_curve(curve, carrier) for carrier in own_carriers)
            assert not any(owned_curve(curve, carrier) for carrier in source_carriers)

    for control_name in ("CTRL_C_jaw", "CTRL_L_mouth_cornerPull", "CTRL_C_eye"):
        pose_bone = new_instance.face_board.pose.bones.get(control_name)
        assert pose_bone.location.length == pytest.approx(0.0, abs=1e-5), (
            f"'{control_name}' kept the expression it was copied from"
        )


def test_duplicated_face_board_follows_its_own_head(animated_scene_with_a_second_character):
    instance = animated_scene_with_a_second_character[1]

    for control_name in ("CTRL_faceGUI", "CTRL_C_eyesAim"):
        pose_bone = instance.face_board.pose.bones[control_name]
        constraints = [constraint for constraint in pose_bone.constraints if constraint.type == "CHILD_OF"]
        assert len(constraints) == 1
        assert constraints[0].target == instance.body_rig
        assert constraints[0].subtarget == "head"


def test_second_import_leaves_the_first_face_board_alone(animated_scene_with_a_second_character):
    """Positioning the new board must not drag the animated one it was copied from."""
    instance = animated_scene_with_a_second_character[0]
    face_board = instance.face_board

    assert tuple(face_board.location) == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert face_board.animation_data.action is not None
