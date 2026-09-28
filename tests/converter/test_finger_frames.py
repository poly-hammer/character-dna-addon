"""Finger fitting preserves anatomical axes, roll, and unchanged conversions."""

import bpy
import numpy as np
import pytest

from mathutils import Euler, Matrix, Vector

from character_dna.editors.converter.finger_frames import (
    fit_finger_orientations,
    hand_bind_transforms,
    skin_hand_positions,
    weighted_rotation,
)
from character_dna.utilities import switch_to_bone_edit_mode, switch_to_object_mode


@pytest.mark.parametrize("scale", [0.7, 1.0, 1.4])
def test_weighted_frame_recovers_roll_and_ignores_translation_scale(scale):
    source = np.array([(0, 0, 0), (1, 0, 0), (0, 0.3, 0), (0.2, 0.1, 0.5)])
    expected = Euler((0.4, -0.3, 0.8)).to_matrix()
    target = scale * source @ np.asarray(expected).T + [3, -7, 2]
    actual = weighted_rotation(source, target, [0.2, 0.4, 0.9, 0.7])
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.determinant() == pytest.approx(1, abs=1e-6)


def test_collinear_samples_do_not_invent_roll():
    points = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    assert weighted_rotation(points, points, [1, 1, 1]) is None


@pytest.mark.parametrize("changed", [False, True])
def test_finger_and_corrective_frames_follow_wrap_without_moving_origins(addon, changed):
    armature = bpy.data.armatures.new("finger_fit_test")
    rig = bpy.data.objects.new("finger_fit_test", armature)
    bpy.context.scene.collection.objects.link(rig)
    mesh = bpy.data.meshes.new("finger_samples")
    obj = bpy.data.objects.new("finger_samples", mesh)
    bpy.context.scene.collection.objects.link(obj)
    try:
        source = np.array([(0, 0, 0), (0.02, 0, 0), (0, 0.01, 0), (0, 0, 0.015)])
        rotation = Euler((0.4, -0.25, 0.7)).to_matrix() if changed else Matrix.Identity(3)
        target = source @ np.asarray(rotation).T
        mesh.from_pydata(target.tolist(), [], [])
        for name in ("index_01_l", "index_02_l", "index_03_l"):
            group = obj.vertex_groups.new(name=name)
            group.add(list(range(4)), 1.0, "REPLACE")
        switch_to_bone_edit_mode(rig)
        for name, location in (
            ("hand_l", (-0.02, 0, 0)),
            ("index_01_l", (0, 0, 0)),
            ("index_02_l", (0.02, 0, 0)),
            ("index_03_l", (0.04, 0, 0)),
            ("index_01_half_l", (0.001, 0.002, 0)),
            ("index_01_mcp_l", (0, 0.003, 0)),
        ):
            bone = armature.edit_bones.new(name)
            bone.head = location
            bone.tail = Vector(location) + Vector((0, 0.01, 0))
        bind = {bone.name: bone.matrix.copy() for bone in armature.edit_bones}
        for bone in armature.edit_bones:
            matrix = bone.matrix.copy()
            matrix.translation = rotation @ matrix.translation
            bone.matrix = matrix
        origins = {bone.name: bone.head.copy() for bone in armature.edit_bones}
        fit_finger_orientations(armature.edit_bones, bind, obj, source)
        for bone in armature.edit_bones:
            expected = bind[bone.name].to_3x3()
            if bone.name.startswith("index"):
                expected = rotation @ expected
            np.testing.assert_allclose(bone.matrix.to_3x3(), expected, atol=1e-6)
            np.testing.assert_array_equal(bone.head, origins[bone.name])
    finally:
        switch_to_object_mode()
        bpy.data.objects.remove(rig, do_unlink=True)
        bpy.data.armatures.remove(armature)
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


@pytest.mark.parametrize("side", ["l", "r"])
def test_hand_alignment_recovers_reference_pose_preserving_shape_and_body(addon, side):
    """A rotated custom hand must transfer back with its lengths and offsets intact."""
    names = ["pelvis", f"hand_{side}", f"index_01_{side}", f"index_02_{side}", f"index_01_half_{side}"]
    parents = dict(zip(names, [None, names[0], names[1], names[2], names[2]], strict=True))
    template = {
        name: Matrix.Translation(Vector(location))
        for name, location in zip(
            names, [(0, 0, 1), (0.4, 0, 1), (0.45, 0, 1), (0.48, 0, 1), (0.45, 0.01, 1)], strict=True
        )
    }
    rotation = Euler((0.5, -0.7, 0.3)).to_matrix().to_4x4()
    wrist = template[names[1]].translation
    shift = Vector((0.07, -0.03, 0.02))
    fitted = {name: matrix.copy() for name, matrix in template.items()}
    fitted["pelvis"].translation += Vector((0, -0.1, 0.03))
    for name in names[1:]:
        matrix = rotation @ template[name]
        matrix.translation = wrist + shift + rotation.to_3x3() @ (1.3 * (template[name].translation - wrist))
        fitted[name] = matrix
    aligned = hand_bind_transforms(template, fitted, parents)
    assert "pelvis" not in aligned
    for name in names[1:]:
        expected = template[name].copy()
        expected.translation = wrist + shift + 1.3 * (template[name].translation - wrist)
        np.testing.assert_allclose(aligned[name], expected, atol=2e-7)

    mesh = bpy.data.meshes.new("hand_alignment_skin")
    obj = bpy.data.objects.new("hand_alignment_skin", mesh)
    try:
        shape = [Vector((0.5, 0.007, 1.008)), Vector((0.46, -0.012, 1.009))]
        shaped = [wrist + shift + 1.3 * (point - wrist) for point in shape]
        wrapped = [wrist + shift + rotation.to_3x3() @ (1.3 * (point - wrist)) for point in shape]
        untouched = Vector((0.012, -0.042, 0.98))
        mesh.from_pydata([*wrapped, untouched], [], [])
        obj.vertex_groups.new(name=names[2]).add([0, 1], 0.3, "REPLACE")
        obj.vertex_groups.new(name=names[3]).add([0, 1], 0.7, "REPLACE")
        obj.vertex_groups.new(name="pelvis").add([2], 1, "REPLACE")
        before = mesh.vertices[2].co.copy()
        skin_hand_positions(obj, fitted, aligned)
        np.testing.assert_allclose([v.co for v in mesh.vertices[:2]], shaped, atol=2e-7)
        np.testing.assert_array_equal(mesh.vertices[2].co, before)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def test_reference_hand_alignment_is_an_exact_noop():
    template = {
        "root": Matrix.Identity(4),
        "hand_l": Matrix.Translation(Vector((0.3, 0.1, 1))),
        "index_01_l": Matrix.Translation(Vector((0.35, 0.1, 1))),
    }
    parents = {"root": None, "hand_l": "root", "index_01_l": "hand_l"}
    assert hand_bind_transforms(template, template, parents) == {}
