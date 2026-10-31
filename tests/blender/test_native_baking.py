"""Baked actions take ownership of their outputs before live evaluation resumes."""

import bpy
import pytest

from mathutils import Quaternion

from character_dna.runtime import controller, engine
from character_dna.utilities import get_active_rig_instance
from character_dna.utilities.action import bake_body_to_action, bake_face_board_to_action
from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigProfile


@pytest.fixture
def native_character(rig_factory, tmp_path):
    with rig_factory.lease(RigProfile(meshes=False, face_board=True, pose_bones=STATE_BONES), tmp_path) as instance:
        yield instance


@pytest.fixture
def animated_character(native_character):
    instance = native_character
    jaw = instance.face_board.pose.bones["CTRL_C_jaw"]
    calf = instance.body_rig.pose.bones["calf_l"]
    for frame, value in ((1, 0.0), (2, 0.8)):
        jaw.location.y = value
        jaw.keyframe_insert("location", frame=frame)
        calf.rotation_quaternion = Quaternion((1, 0, 0, 0)) if frame == 1 else Quaternion((0.5, 0, 0, -0.8660254))
        calf.keyframe_insert("rotation_quaternion", frame=frame)
    bpy.context.scene.frame_set(1)
    return instance


def _pose(instance, component):
    rig = getattr(instance, f"{component}_rig")
    evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
    name = "FACIAL_C_Jaw" if component == "head" else "calf_knee_l"
    return tuple(value for row in evaluated.pose.bones[name].matrix for value in row)


@pytest.mark.parametrize("component", ["head", "body"])
@pytest.mark.parametrize("use_operator", [False, True], ids=["utility", "operator"])
def test_bake_hands_outputs_to_action_and_keeps_other_component_live(animated_character, component, use_operator):
    instance = animated_character
    other = "body" if component == "head" else "head"
    expected = {}
    for frame in (1, 2):
        bpy.context.scene.frame_set(frame)
        expected[frame] = {name: _pose(instance, name) for name in (component, other)}
    assert expected[1][component] != pytest.approx(expected[2][component], abs=1e-5)
    bpy.context.scene.frame_set(1)

    options = {
        "action_name": "Baked",
        "replace_action": False,
        "start_frame": 1,
        "end_frame": 2,
        "clean_curves": False,
        "masks": False,
        "shape_keys": False,
    }
    if use_operator:
        if component == "head":
            result = bpy.ops.character_dna.bake_face_board_animation(**options)
        else:
            result = bpy.ops.character_dna.bake_component_animation(component_type="body", **options)
        assert result == {"FINISHED"}
    else:
        bake = bake_face_board_to_action if component == "head" else bake_body_to_action
        bake(instance=instance, armature_object=getattr(instance, f"{component}_rig"), **options)

    assert instance.auto_evaluate
    assert not getattr(instance, f"auto_evaluate_{component}")
    assert getattr(instance, f"auto_evaluate_{other}")
    assert not controller.is_suspended(instance)
    assert component not in {carrier["component"] for carrier in engine.carriers(instance)}
    assert other in {carrier["component"] for carrier in engine.carriers(instance)}
    assert engine.binding_issues(instance) == []
    assert _pose(instance, component) == pytest.approx(expected[1][component], abs=1e-5)
    controller.reconcile()
    for frame in (2, 1):
        bpy.context.scene.frame_set(frame)
        for name in (component, other):
            assert _pose(instance, name) == pytest.approx(expected[frame][name], abs=1e-5)


@pytest.mark.parametrize("component", ["head", "body"])
def test_bake_without_source_keeps_live_evaluation(animated_character, component):
    instance = animated_character
    source = instance.face_board if component == "head" else instance.body_rig
    source.animation_data.action = None
    bake = bake_face_board_to_action if component == "head" else bake_body_to_action
    bake(
        instance=instance,
        armature_object=getattr(instance, f"{component}_rig"),
        action_name="NoSource",
        replace_action=False,
        start_frame=1,
        end_frame=2,
    )
    assert getattr(instance, f"auto_evaluate_{component}")
    assert engine.active(instance)
    assert engine.binding_issues(instance) == []
    assert not controller.is_suspended(instance)
    assert bpy.data.actions.get("NoSource") is None


@pytest.mark.parametrize("component", ["head", "body"])
def test_rebake_replaces_existing_action_and_stale_driven_curves(animated_character, component):
    from character_dna.utilities.action import get_channel_bag

    instance = animated_character
    rig = getattr(instance, f"{component}_rig")
    if component == "body":
        # An imported animation can contain obsolete corrective curves. Baking
        # must replace those with RigLogic's result, not retain the old keys.
        bag = get_channel_bag(rig.animation_data.action, rig.name)
        curve = bag.fcurves.new(data_path='pose.bones["calf_knee_l"].location', index=0)
        curve.keyframe_points.insert(1, 99.0)
    expected = {}
    for frame in (1, 2):
        bpy.context.scene.frame_set(frame)
        expected[frame] = _pose(instance, component)
    bpy.context.scene.frame_set(1)
    previous = bpy.data.actions.new("Rebaked")
    previous["obsolete"] = True
    bake = bake_face_board_to_action if component == "head" else bake_body_to_action
    bake(
        instance=instance,
        armature_object=rig,
        action_name="Rebaked",
        replace_action=True,
        start_frame=1,
        end_frame=2,
        clean_curves=False,
        masks=False,
        shape_keys=False,
    )
    assert rig.animation_data.action == bpy.data.actions["Rebaked"]
    assert not any(action.get("obsolete") for action in bpy.data.actions)
    for frame in (2, 1):
        bpy.context.scene.frame_set(frame)
        assert _pose(instance, component) == pytest.approx(expected[frame], abs=1e-5)


@pytest.mark.parametrize("component", ["head", "body"])
def test_authoring_resume_preserves_disabled_outputs(animated_character, component):
    instance = animated_character
    setattr(instance, f"auto_evaluate_{component}", False)
    controller.reconcile()
    rig = getattr(instance, f"{component}_rig")
    name = "FACIAL_C_Jaw" if component == "head" else "calf_knee_l"
    bone = rig.pose.bones[name]
    bone.location.y = 0.125
    with controller.authoring_operation(instance):
        pass
    assert bone.location.y == pytest.approx(0.125)
    assert engine.active(instance)
    assert not controller.is_suspended(instance)


@pytest.mark.parametrize("first", ["head", "body"])
def test_baking_both_components_preserves_playback_after_reload(animated_character, first, tmp_path):
    instance = animated_character
    expected = {}
    for frame in (1, 2):
        bpy.context.scene.frame_set(frame)
        expected[frame] = {name: _pose(instance, name) for name in ("head", "body")}
    bpy.context.scene.frame_set(1)
    for component in (first, "body" if first == "head" else "head"):
        bake = bake_face_board_to_action if component == "head" else bake_body_to_action
        bake(
            instance=instance,
            armature_object=getattr(instance, f"{component}_rig"),
            action_name=f"Baked_{component}",
            replace_action=False,
            start_frame=1,
            end_frame=2,
            masks=False,
            shape_keys=False,
        )
    assert not instance.auto_evaluate_head and not instance.auto_evaluate_body
    assert not engine.carriers(instance)
    assert not controller.is_suspended(instance)
    for component in ("head", "body"):
        assert _pose(instance, component) == pytest.approx(expected[1][component], abs=1e-5)

    path = str(tmp_path / "baked.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    instance = get_active_rig_instance()
    controller.reconcile()
    assert not instance.auto_evaluate_head and not instance.auto_evaluate_body
    assert not engine.carriers(instance)
    for frame in (2, 1):
        bpy.context.scene.frame_set(frame)
        for component in ("head", "body"):
            assert _pose(instance, component) == pytest.approx(expected[frame][component], abs=1e-5)
