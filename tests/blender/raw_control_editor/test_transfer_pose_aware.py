"""Pose-aware Transfer Vertices.

When the rig is posed, the source head mesh is displayed through its
armature modifier, so writing the target's absolute shape straight onto the
base vertices double-deforms it (the lip distortion in the user's repro).
The transfer instead moves the base by ``target - deformed`` so the
*displayed* (armature-deformed) mesh lands on the target's absolute shape.

These tests build a mesh rigidly bound to a single translated bone (a
pure-translation pose has an identity rotation part, so the delta transfer
is exact) and verify the evaluated/deformed geometry ends up on the target.
"""

from __future__ import annotations

import bpy
import numpy as np
import pytest

from character_dna.editors.raw_control_editor.utilities import (
    _read_evaluated_mesh_vertices_local,
    _read_mesh_vertices_local,
    _write_mesh_vertices_masked,
)


_VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
_FACE = [0, 1, 2, 3]
_TARGET_OFFSET = np.array([0.1, 0.4, -0.2])


@pytest.fixture
def posed_mesh():
    """A 4-vert mesh rigidly bound (weight 1) to a single bone that is
    translated, so the depsgraph-evaluated mesh is deformed away from its
    base vertices."""
    created: list[bpy.types.Object] = []
    arm_data = bpy.data.armatures.new("tpa_arm")
    arm_obj = bpy.data.objects.new("tpa_arm", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    created.append(arm_obj)

    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bone = arm_data.edit_bones.new("b0")
    edit_bone.head = (0.0, 0.0, 0.0)
    edit_bone.tail = (0.0, 1.0, 0.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = bpy.data.meshes.new("tpa_mesh")
    mesh.from_pydata(list(_VERTS), [], [_FACE])
    mesh.update()
    mesh_obj = bpy.data.objects.new("tpa_mesh", mesh)
    bpy.context.scene.collection.objects.link(mesh_obj)
    created.append(mesh_obj)

    group = mesh_obj.vertex_groups.new(name="b0")
    group.add(list(range(len(mesh.vertices))), 1.0, "REPLACE")
    modifier = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = arm_obj  # type: ignore[attr-defined]

    # Pure-translation pose (no rotation): the per-vertex skinning matrix is a
    # pure translation, so its rotation part is identity and the delta transfer
    # is exact.
    arm_obj.pose.bones["b0"].location = (0.3, -0.2, 0.15)

    view_layer.objects.active = prev_active
    bpy.context.view_layer.update()

    yield mesh_obj

    for obj in created:
        bpy.data.objects.remove(obj, do_unlink=True)
    if arm_data.users == 0:
        bpy.data.armatures.remove(arm_data)


def test_evaluated_read_returns_deformed_geometry(posed_mesh):
    base = np.asarray(_read_mesh_vertices_local(posed_mesh), dtype=np.float64)
    deformed = _read_evaluated_mesh_vertices_local(posed_mesh)
    assert deformed.shape == base.shape
    # The bone was translated, so the deformed geometry differs from the base.
    assert not np.allclose(deformed, base)


def test_pose_aware_transfer_lands_display_on_target(posed_mesh):
    base = np.asarray(_read_mesh_vertices_local(posed_mesh), dtype=np.float64)
    deformed = _read_evaluated_mesh_vertices_local(posed_mesh)
    target = base + _TARGET_OFFSET

    # Pose-aware write: base + (target - deformed).
    new_positions = (base + (target - deformed)).tolist()
    _write_mesh_vertices_masked(posed_mesh, new_positions, mask=None, strength=1.0)
    bpy.context.view_layer.update()

    # The displayed (armature-deformed) mesh now matches the target absolute
    # shape (exact for a pure-translation pose).
    display = _read_evaluated_mesh_vertices_local(posed_mesh)
    np.testing.assert_allclose(display, target, atol=1e-4)


def test_naive_write_double_deforms_when_posed(posed_mesh):
    # Contrast: writing the target straight onto the base (the old behavior)
    # leaves the displayed mesh offset from the target by the pose.
    base = np.asarray(_read_mesh_vertices_local(posed_mesh), dtype=np.float64)
    target = base + _TARGET_OFFSET
    _write_mesh_vertices_masked(posed_mesh, target.tolist(), mask=None, strength=1.0)
    bpy.context.view_layer.update()
    display = _read_evaluated_mesh_vertices_local(posed_mesh)
    assert not np.allclose(display, target, atol=1e-3)
