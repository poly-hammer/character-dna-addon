"""Display basis transforms reproduce Blender's parent-local matrix contract."""

import bpy
import numpy as np
import pytest

from mathutils import Euler, Matrix, Vector

from character_dna.editors.converter.core import compute_pose_basis_targets
from character_dna.utilities import switch_to_bone_edit_mode, switch_to_object_mode


@pytest.mark.parametrize("singular", [False, True])
def test_native_rest_endpoints_match_blender_object_space_rounding(singular):
    from character_dna.bindings import load_converter_core

    source = Matrix.LocRotScale(Vector((0.1, -0.3, 0.7)), Euler((0.2, -0.1, 0.5)), Vector((1.1, 0.9, 1.2)))
    destination = Matrix.LocRotScale(
        Vector((-0.2, 0.4, -0.5)), Euler((-0.3, 0.4, 0.1)), Vector((0 if singular else 0.8, 1.2, 0.9))
    )
    points = np.array([[0.13, 0.21, -0.1], [0.25, -0.4, 0.5], [0, 0, 0]], dtype=np.float64)
    expected = [destination.inverted_safe() @ (source @ Vector(point)) for point in points]
    values = load_converter_core().joint_rest_endpoints(
        points,
        np.array(source, dtype=np.float64).reshape(1, 16),
        np.array(destination, dtype=np.float64).reshape(1, 16),
    )
    result = np.frombuffer(values, dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("parent_present", [False, True])
@pytest.mark.parametrize("degenerate", [False, True])
def test_pose_basis_matches_parent_local_frames_and_safe_inverse(parent_present, degenerate):
    data = bpy.data.armatures.new("basis contract")
    rig = bpy.data.objects.new("basis contract", data)
    bpy.context.scene.collection.objects.link(rig)
    try:
        switch_to_bone_edit_mode(rig)
        parent = data.edit_bones.new("parent")
        parent.head, parent.tail = (0, 0, 0), (0, 1, 0)
        child = data.edit_bones.new("child")
        child.head, child.tail, child.parent = (0, 1, 0), (0, 2, 0), parent
        switch_to_object_mode()
        bind = {
            "child": Matrix.LocRotScale(Vector((0.1, 0.3, -0.2)), Euler((0.3, -0.2, 0.4)), Vector((1, 1, 1))),
        }
        fitted = {
            "child": Matrix.LocRotScale(Vector((0.2, 0.5, -0.4)), Euler((-0.4, 0.5, 0.1)), Vector((1, 1, 1))),
        }
        if degenerate:
            bind["child"] = Matrix.Diagonal(Vector((0, 1, 1, 1)))
        if parent_present:
            bind["parent"] = Matrix.Translation((0.2, 0.3, 0.4))
            fitted["parent"] = Matrix.Rotation(0.3, 4, "Z")
            expected = (bind["parent"].inverted_safe() @ bind["child"]).inverted_safe() @ (
                fitted["parent"].inverted_safe() @ fitted["child"]
            )
        else:
            expected = bind["child"].inverted_safe() @ fitted["child"]
        result = compute_pose_basis_targets(rig, bind, fitted)
        np.testing.assert_allclose(result["child"], expected, atol=1e-6, rtol=1e-6)
        assert set(result) == set(bind)
    finally:
        switch_to_object_mode()
        bpy.data.objects.remove(rig, do_unlink=True)
        bpy.data.armatures.remove(data)
