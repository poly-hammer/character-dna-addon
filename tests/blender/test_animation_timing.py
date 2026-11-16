"""Fractional FBX times survive import; rounding changes times, not sample values."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.fbx.reader import FbxAnimationClip
from character_dna.fbx.writer import get_channel_container
from character_dna.utilities.action import import_action_from_fbx, import_face_board_action_from_fbx


@pytest.fixture
def timing_rig():
    data = bpy.data.armatures.new("Timing")
    obj = bpy.data.objects.new("Timing", data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = data.edit_bones.new("CTRL_C_jaw")
    bone.head, bone.tail = (0, 0, 0), (0, 0, 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.pose.bones["CTRL_C_jaw"].rotation_mode = "QUATERNION"
    try:
        yield obj
    finally:
        action = obj.animation_data.action if obj.animation_data else None
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.armatures.remove(data)
        if action is not None:
            bpy.data.actions.remove(action)


def source_clip():
    frames = np.array([0.0, 0.25, 1.0, 1.25, 2.75])
    translations = np.zeros((5, 1, 3))
    translations[:, 0, 0] = [0, 1, 2, 3, 4]
    return FbxAnimationClip(
        node_names=("CTRL_C_jaw",),
        parent_indices=np.array([-1]),
        rest_rotations=np.array([[1, 0, 0, 0]]),
        rest_translations=np.zeros((1, 3)),
        rest_world_rotations=np.array([[1, 0, 0, 0]]),
        rest_world_translations=np.zeros((1, 3)),
        rotations=np.tile([1, 0, 0, 0], (5, 1, 1)),
        translations=translations,
        animated_node_names=frozenset({"CTRL_C_jaw"}),
        frame_rate=30.0,
        file_frame_rate=30.0,
        unit_meters=1.0,
        up_axis="Z",
        take_name="Fractional source",
        sample_times=frames / 30.0,
    )


@pytest.mark.parametrize("face_board", [False, True], ids=["skeleton", "face_board"])
@pytest.mark.parametrize("rounded", [None, False, True], ids=["default", "fractional", "rounded"])
@pytest.mark.parametrize(
    ("fps", "fps_base", "match_rate", "fractional", "whole"),
    [
        (24, 1.0, True, [1, 1.2, 1.8, 2, 3.2], [1, 2, 3]),
        (30, 1.001, True, [1, 1.24975025, 1.999001, 2.24875125, 3.74725275], [1, 2, 4]),
        (24, 1.0, False, [1, 1.25, 2, 2.25, 3.75], [1, 2, 4]),
    ],
    ids=["30_to_24", "ntsc_scene", "source_frames"],
)
def test_import_timing_and_collisions(timing_rig, face_board, rounded, fps, fps_base, match_rate, fractional, whole):
    scene = bpy.context.scene
    previous_rate = scene.render.fps, scene.render.fps_base
    scene.render.fps, scene.render.fps_base = fps, fps_base
    options = {"match_frame_rate": match_rate, "clip": source_clip()}
    if rounded is not None:
        options["round_sub_frames"] = rounded
    try:
        if face_board:
            action = import_face_board_action_from_fbx(
                SimpleNamespace(name="Timing"), "timing.fbx", timing_rig, **options
            )
        else:
            action = import_action_from_fbx(SimpleNamespace(name="Timing"), "timing.fbx", "body", timing_rig, **options)
    finally:
        scene.render.fps, scene.render.fps_base = previous_rate

    curves = get_channel_container(action, timing_rig).fcurves
    location = curves.find('pose.bones["CTRL_C_jaw"].location', index=0)
    expected_frames = whole if rounded else fractional
    for curve in curves:
        np.testing.assert_allclose([key.co.x for key in curve.keyframe_points], expected_frames, rtol=0, atol=1e-6)
    expected_values = [1, 3, 4] if rounded else [0, 1, 2, 3, 4]
    np.testing.assert_allclose([key.co.y for key in location.keyframe_points], expected_values, rtol=0, atol=1e-6)
    for frame, value in zip(expected_frames, expected_values, strict=True):
        assert location.evaluate(frame) == pytest.approx(value, abs=1e-5)
    for index in range(len(expected_frames) - 1):
        midpoint = (expected_frames[index] + expected_frames[index + 1]) / 2
        value = (expected_values[index] + expected_values[index + 1]) / 2
        assert location.evaluate(midpoint) == pytest.approx(value, abs=1e-5)


@pytest.mark.parametrize("rounded", [False, True])
def test_confirmation_keeps_rounding_choice(monkeypatch, rounded):
    from character_dna import operators

    calls = []

    def import_animation(*args, **kwargs):
        calls.append(kwargs)
        return {"FINISHED"}

    fake_ops = SimpleNamespace(
        **{operators.ToolInfo.NAME: SimpleNamespace(import_component_animation=import_animation)}
    )
    monkeypatch.setattr(operators, "bpy", SimpleNamespace(ops=fake_ops))
    confirmation = SimpleNamespace(
        operator_name="import_component_animation",
        filepath="timing.fbx",
        component_type="head",
        round_sub_frames=rounded,
        match_frame_rate=True,
        prefix_instance_name=True,
        prefix_component_name=True,
    )
    assert operators.ConfirmAnimationImport.execute(confirmation, None) == {"FINISHED"}
    assert calls[0]["round_sub_frames"] is rounded
    assert calls[0]["component_type"] == "head"
    assert calls[0]["ignore_validation"] is True
