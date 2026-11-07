"""Native output attributes agree with Blender's scene boundary."""

import bpy
import numpy as np
import pytest

from mathutils import Euler

from character_dna.bindings import load_converter_core
from character_dna.dna_io.exporter import DNAExporter
from character_dna.editors.converter.native_joints import _matrices
from fixtures.rigs import RigProfile


@pytest.mark.parametrize("component", ["head", "body"])
def test_native_channels_and_smooth_normals_match_scene_capture(rig_factory, tmp_path, component):
    profile = RigProfile(components=(component,), lods=(0,), shape_keys=False, face_board=False, materials=False)
    native = load_converter_core()
    with rig_factory.lease(profile, tmp_path / "scene") as instance:
        rig = getattr(instance, f"{component}_rig")
        _, names, _, _, translations, rotations = DNAExporter.get_bone_transforms(rig, extra_bones=[])
        actual_t, actual_r = native.joint_channels(
            _matrices({bone.name: bone.matrix_local for bone in rig.data.bones}),
            names,
            {bone.name: bone.parent.name if bone.parent else None for bone in rig.data.bones},
        )
        actual_t = np.frombuffer(actual_t, dtype=np.float32).reshape(-1, 3)
        actual_r = np.frombuffer(actual_r, dtype=np.float32).reshape(-1, 3)
        np.testing.assert_allclose(actual_t, translations, atol=1e-4, rtol=0)
        # Euler winding is nonunique at singular orientations; compare rotations.
        actual_m = [Euler(np.radians(row), "XZY").to_matrix() for row in actual_r]
        expected_m = [Euler(np.radians(row), "XZY").to_matrix() for row in rotations]
        np.testing.assert_allclose(actual_m, expected_m, atol=2e-5, rtol=0)

        mesh = getattr(instance, f"{component}_mesh").data
        # Converter imports deliberately omit authored normals.
        if mesh.has_custom_normals:
            mesh.normals_split_custom_set([(0.0, 0.0, 0.0)] * len(mesh.loops))
        positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", positions.ravel())
        positions[:, 0] *= 1.17
        positions[:, 2] += 0.03 * np.sin(positions[:, 0] * 7)
        # Use a fresh mesh to avoid the imported custom-normal layer entirely.
        fresh = bpy.data.meshes.new("native-smooth-parity")
        try:
            fresh.from_pydata(positions, [], [list(poly.vertices) for poly in mesh.polygons])
            fresh.polygons.foreach_set("use_smooth", np.ones(len(fresh.polygons), dtype=bool))
            fresh.update()
            corners = np.empty(len(fresh.loops), dtype=np.int32)
            sizes = np.empty(len(fresh.polygons), dtype=np.int32)
            fresh.loops.foreach_get("vertex_index", corners)
            fresh.polygons.foreach_get("loop_total", sizes)
            expected = np.empty((len(fresh.loops), 3), dtype=np.float32)
            fresh.corner_normals.foreach_get("vector", expected.ravel())
            actual = np.frombuffer(native.mesh_smooth_normals(positions, corners, sizes), dtype=np.float32).reshape(
                -1, 3
            )
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
        finally:
            bpy.data.meshes.remove(fresh)


def test_native_channel_euler_orders_and_gimbal_limits():
    native = load_converter_core()
    rng = np.random.default_rng(408)
    angles = [*rng.uniform(-np.pi, np.pi, (200, 3)), [0.3, 0.2, np.pi / 2], [-0.3, -0.2, -np.pi / 2]]
    frames = {str(i): Euler(row, "XZY").to_matrix().to_4x4() for i, row in enumerate(angles)}
    _, rotations = native.joint_channels(_matrices(frames), list(frames), dict.fromkeys(frames))
    rotations = np.frombuffer(rotations, dtype=np.float32).reshape(-1, 3)
    expected = [frame.to_3x3() for frame in frames.values()]
    actual = [Euler(np.radians(row), "XZY").to_matrix() for row in rotations]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
