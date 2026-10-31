"""Regression tests for shape-key-aware mesh writes in the Raw Control
Editor.

Bug: ``bpy.ops.character_dna.rce_transfer_target_to_head`` wrote the
sculpted vertices onto ``mesh.vertices.co`` only. When the head mesh has
imported blend shapes, Blender drives the displayed geometry from the
shape-key data blocks (the reference/``Basis`` key), so the bare
``mesh.vertices`` write was invisible and the head appeared unchanged.

The fix propagates the per-vertex offset to every shape key block so the
rest pose (reference key) updates while each corrective's delta relative
to the basis is preserved.
"""

from __future__ import annotations

import bpy
import numpy as np

from character_dna.constants import SHAPE_KEY_BASIS_NAME
from character_dna.editors.raw_control_editor.utilities import _write_mesh_vertices_masked


_QUAD_FACE = [0, 1, 2, 3]
_BASE_POSITIONS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
]


def _read_co(data: object, count: int) -> np.ndarray:
    flat = np.zeros(count * 3, dtype=np.float32)
    data.foreach_get("co", flat)  # type: ignore[attr-defined]
    return flat.reshape(count, 3)


def _make_quad_with_shape_keys(name: str) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(_BASE_POSITIONS, [], [_QUAD_FACE])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    # Basis (reference) key plus one corrective with a known delta.
    obj.shape_key_add(name=SHAPE_KEY_BASIS_NAME, from_mix=False)
    corrective = obj.shape_key_add(name="corrective", from_mix=False)
    # Push the corrective's first vertex out by a fixed delta.
    corrective.data[0].co.z += 0.5
    return obj


def test_masked_write_updates_basis_and_preserves_corrective_delta():
    obj = _make_quad_with_shape_keys("head_with_keys")
    try:
        mesh = obj.data
        count = len(mesh.vertices)
        shape_keys = mesh.shape_keys
        basis = shape_keys.key_blocks[SHAPE_KEY_BASIS_NAME]
        corrective = shape_keys.key_blocks["corrective"]

        # Corrective delta (relative to basis) captured before the write.
        delta_before = _read_co(corrective.data, count) - _read_co(basis.data, count)

        # New rest-pose geometry: shift every vertex +1 on Y.
        new_positions = [(x, y + 1.0, z) for (x, y, z) in _BASE_POSITIONS]
        written = _write_mesh_vertices_masked(obj, new_positions, mask=None, strength=1.0)
        assert written == count

        expected = np.asarray(new_positions, dtype=np.float32)

        # The bare mesh vertices were updated (DNA export reads these).
        np.testing.assert_allclose(_read_co(mesh.vertices, count), expected, atol=1e-6)
        # The reference/Basis key was updated -- this is what Blender
        # displays for the rest pose, so the head now visibly changes.
        np.testing.assert_allclose(_read_co(basis.data, count), expected, atol=1e-6)
        # The corrective rode along: its delta relative to the basis is
        # preserved (so locked correctives are not silently distorted).
        delta_after = _read_co(corrective.data, count) - _read_co(basis.data, count)
        np.testing.assert_allclose(delta_after, delta_before, atol=1e-6)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_masked_write_without_shape_keys_unaffected():
    mesh = bpy.data.meshes.new("plain")
    mesh.from_pydata(_BASE_POSITIONS, [], [_QUAD_FACE])
    mesh.update()
    obj = bpy.data.objects.new("plain", mesh)
    bpy.context.scene.collection.objects.link(obj)
    try:
        new_positions = [(x, y + 2.0, z) for (x, y, z) in _BASE_POSITIONS]
        _write_mesh_vertices_masked(obj, new_positions, mask=None, strength=1.0)
        expected = np.asarray(new_positions, dtype=np.float32)
        np.testing.assert_allclose(_read_co(mesh.vertices, len(mesh.vertices)), expected, atol=1e-6)
        assert mesh.shape_keys is None
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_masked_write_partial_selection_shifts_only_selected_vertices():
    obj = _make_quad_with_shape_keys("head_partial")
    try:
        mesh = obj.data
        count = len(mesh.vertices)
        basis = mesh.shape_keys.key_blocks[SHAPE_KEY_BASIS_NAME]
        basis_before = _read_co(basis.data, count)

        # Only vertex 0 is in the write mask.
        mask = np.zeros(count, dtype=bool)
        mask[0] = True
        new_positions = [(x, y, z + 3.0) for (x, y, z) in _BASE_POSITIONS]
        _write_mesh_vertices_masked(obj, new_positions, mask=mask, strength=1.0)

        basis_after = _read_co(basis.data, count)
        # Vertex 0 moved; all others untouched on the basis key.
        assert not np.allclose(basis_after[0], basis_before[0])
        np.testing.assert_allclose(basis_after[1:], basis_before[1:], atol=1e-6)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
