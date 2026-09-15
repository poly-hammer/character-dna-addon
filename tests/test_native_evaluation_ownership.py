"""Automatic drivers release editable channels; explicit samples still use native RigLogic."""

import bpy
import pytest

from mathutils import Quaternion

from character_dna.runtime import controller, engine
from character_dna.utilities import get_active_rig_instance
from constants import TEST_DNA_FOLDER


@pytest.fixture
def native_character(addon):
    bpy.ops.wm.read_homefile(app_template="")
    bpy.ops.character_dna.import_dna(
        filepath=str(TEST_DNA_FOLDER / "ada" / "head.dna"),
        include_body=True,
        import_face_board=True,
        import_mesh=False,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
    )
    return get_active_rig_instance()


def pose_matrix(rig, name):
    evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return tuple(value for row in evaluated.pose.bones[name].matrix for value in row)


@pytest.mark.parametrize("flag", ["auto_evaluate", "auto_evaluate_head", "auto_evaluate_body"])
def test_disabled_outputs_stay_editable_without_waiting_for_timer(native_character, flag):
    instance = native_character
    component = "head" if flag == "auto_evaluate_head" else "body"
    name = "FACIAL_C_Jaw" if component == "head" else "calf_knee_l"
    rig = getattr(instance, f"{component}_rig")
    setattr(instance, flag, False)
    bone = rig.pose.bones[name]
    bone.location.y = 0.125
    rig.update_tag()
    bpy.context.view_layer.update()
    assert bone.location.y == pytest.approx(0.125)
    controller.reconcile()
    bpy.context.scene.frame_set(2)
    assert bone.location.y == pytest.approx(0.125)


@pytest.mark.parametrize("flag", ["auto_evaluate", "auto_evaluate_body"])
def test_explicit_body_sample_matches_live_pose_without_reenabling_auto(native_character, flag):
    instance = native_character
    rig = instance.body_rig
    driver = rig.pose.bones["calf_l"]
    rotation = Quaternion((0.5, 0.0, 0.0, -0.8660254))
    driver.rotation_quaternion = rotation
    rig.update_tag()
    instance.evaluate(component="body")
    expected = pose_matrix(rig, "calf_knee_l")
    driver.rotation_quaternion = Quaternion()
    rig.update_tag()
    bpy.context.view_layer.update()
    assert pose_matrix(rig, "calf_knee_l") != pytest.approx(expected, abs=1e-5)
    setattr(instance, flag, False)
    controller.reconcile()
    driver.rotation_quaternion = rotation
    rig.update_tag()
    bpy.context.view_layer.update()
    assert pose_matrix(rig, "calf_knee_l") != pytest.approx(expected, abs=1e-5)
    instance.evaluate(component="body")
    assert pose_matrix(rig, "calf_knee_l") == pytest.approx(expected, abs=1e-5)
    assert not getattr(instance, flag)
    rig.pose.bones["calf_knee_l"].location.y = 0.125
    bpy.context.scene.frame_set(3)
    assert rig.pose.bones["calf_knee_l"].location.y == pytest.approx(0.125)


def test_body_toggle_keeps_head_live_and_body_resumes(native_character):
    instance = native_character
    rig = instance.body_rig
    instance.auto_evaluate_body = False
    before_head = pose_matrix(instance.head_rig, "FACIAL_C_Jaw")
    instance.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.8
    instance.face_board.update_tag()
    bpy.context.view_layer.update()
    assert pose_matrix(instance.head_rig, "FACIAL_C_Jaw") != pytest.approx(before_head, abs=1e-5)
    rig.pose.bones["calf_l"].rotation_quaternion = Quaternion((0.5, 0, 0, -0.8660254))
    rig.update_tag()
    instance.evaluate(component="body")
    expected = pose_matrix(rig, "calf_knee_l")
    rig.pose.bones["calf_knee_l"].location.y = 0.125
    bpy.context.view_layer.update()
    instance.auto_evaluate_body = True
    controller.reconcile()
    assert pose_matrix(rig, "calf_knee_l") == pytest.approx(expected, abs=1e-5)


def test_enabling_component_omitted_from_rebuild_installs_it(native_character):
    instance = native_character
    instance.auto_evaluate_body = False
    controller.rebuild(instance)
    assert {carrier["component"] for carrier in engine.carriers(instance)} == {"head", "switches"}
    instance.auto_evaluate_body = True
    controller.reconcile()
    assert engine.binding_issues(instance) == []
    assert {carrier["component"] for carrier in engine.carriers(instance)} == {"head", "body", "switches"}


def test_explicit_head_sample_leaves_disabled_body_edits_alone(native_character):
    instance = native_character
    face = instance.face_board
    face.pose.bones["CTRL_C_jaw"].location.y = 0.8
    face.update_tag()
    instance.evaluate(component="head")
    expected = pose_matrix(instance.head_rig, "FACIAL_C_Jaw")
    face.pose.bones["CTRL_C_jaw"].location.y = 0.0
    face.update_tag()
    bpy.context.view_layer.update()
    instance.auto_evaluate_head = False
    instance.auto_evaluate_body = False
    instance.body_rig.pose.bones["calf_knee_l"].location.y = 0.125
    face.pose.bones["CTRL_C_jaw"].location.y = 0.8
    face.pose.bones["CTRL_faceGUIfollowHead"].location.y = 0.0
    face.update_tag()
    instance.evaluate(component="head")
    assert pose_matrix(instance.head_rig, "FACIAL_C_Jaw") == pytest.approx(expected, abs=1e-5)
    assert instance.body_rig.pose.bones["calf_knee_l"].location.y == pytest.approx(0.125)
    constraint = next(item for item in face.pose.bones["CTRL_faceGUI"].constraints if item.type == "CHILD_OF")
    assert constraint.influence == 0.0
    assert not instance.auto_evaluate_head and not instance.auto_evaluate_body


def test_component_ownership_survives_save_reload(native_character, tmp_path):
    instance = native_character
    rig = instance.body_rig
    rig.pose.bones["calf_l"].rotation_quaternion = Quaternion((0.5, 0, 0, -0.8660254))
    rig.update_tag()
    instance.evaluate(component="body")
    expected = pose_matrix(rig, "calf_knee_l")
    instance.auto_evaluate_body = False
    rig.pose.bones["calf_knee_l"].location.y = 0.125
    bpy.context.view_layer.update()
    path = str(tmp_path / "disabled_body.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    instance = get_active_rig_instance()
    assert not instance.auto_evaluate_body
    assert instance.body_rig.pose.bones["calf_knee_l"].location.y == pytest.approx(0.125)
    instance.auto_evaluate_body = True
    bpy.context.view_layer.update()
    assert pose_matrix(instance.body_rig, "calf_knee_l") == pytest.approx(expected, abs=1e-5)
