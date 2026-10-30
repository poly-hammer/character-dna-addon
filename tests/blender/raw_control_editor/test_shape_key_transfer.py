"""Tests for the Raw Control Editor's shape-key transfer + locking helpers.

Covers the masked shape-key block write used by "Transfer Shape to Shape Key"
and the session-scoped locking that keeps every head shape key locked except the
one driven by the active raw control (and the Basis).
"""

from __future__ import annotations

from types import SimpleNamespace

import bpy
import numpy as np

from character_dna.constants import SHAPE_KEY_BASIS_NAME
from character_dna.editors.raw_control_editor.utilities import (
    _write_shape_key_block_masked,
    lock_all_head_shape_keys_except_basis,
    lock_head_shape_keys,
)


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


def _make_quad(name: str, key_block_names: list[str]) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(_BASE_POSITIONS, [], [_QUAD_FACE])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name=SHAPE_KEY_BASIS_NAME, from_mix=False)
    for block_name in key_block_names:
        obj.shape_key_add(name=block_name, from_mix=False)
    return obj


class _FakeInstance:
    """Minimal stand-in exposing the surface the locking helpers read."""

    def __init__(self, name: str, meshes: dict[int, bpy.types.Object]) -> None:
        self.name = name
        self.edit_target = SimpleNamespace(dna_reader=object(), mesh_index_lookup=meshes)


# ---------------------------------------------------------------------------
# _write_shape_key_block_masked
# ---------------------------------------------------------------------------
def test_block_write_full_strength_sets_target() -> None:
    obj = _make_quad("blk_full", ["head_lod0__jaw_open"])
    try:
        count = len(obj.data.vertices)
        block = obj.data.shape_keys.key_blocks["head_lod0__jaw_open"]
        new_positions = [(x, y, z + 1.0) for (x, y, z) in _BASE_POSITIONS]
        written = _write_shape_key_block_masked(obj, block, new_positions, mask=None, strength=1.0)
        assert written == count
        np.testing.assert_allclose(_read_co(block.data, count), np.asarray(new_positions, dtype=np.float32), atol=1e-6)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_block_write_half_strength_is_midpoint() -> None:
    obj = _make_quad("blk_half", ["head_lod0__jaw_open"])
    try:
        count = len(obj.data.vertices)
        block = obj.data.shape_keys.key_blocks["head_lod0__jaw_open"]
        before = _read_co(block.data, count)
        new_positions = [(x, y, z + 2.0) for (x, y, z) in _BASE_POSITIONS]
        _write_shape_key_block_masked(obj, block, new_positions, mask=None, strength=0.5)
        expected = before + 0.5 * (np.asarray(new_positions, dtype=np.float32) - before)
        np.testing.assert_allclose(_read_co(block.data, count), expected, atol=1e-6)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_block_write_mask_limits_to_selected_vertices() -> None:
    obj = _make_quad("blk_mask", ["head_lod0__jaw_open"])
    try:
        count = len(obj.data.vertices)
        block = obj.data.shape_keys.key_blocks["head_lod0__jaw_open"]
        before = _read_co(block.data, count)
        mask = np.zeros(count, dtype=bool)
        mask[0] = True
        new_positions = [(x, y, z + 3.0) for (x, y, z) in _BASE_POSITIONS]
        written = _write_shape_key_block_masked(obj, block, new_positions, mask=mask, strength=1.0)
        assert written == 1
        after = _read_co(block.data, count)
        assert not np.allclose(after[0], before[0])
        np.testing.assert_allclose(after[1:], before[1:], atol=1e-6)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# shape-key locking
# ---------------------------------------------------------------------------
def test_lock_head_shape_keys_keeps_active_and_basis_unlocked() -> None:
    obj = _make_quad("lock_active", ["head_lod0__jaw_open", "head_lod0__brow_down"])
    try:
        instance = _FakeInstance("Ada", {0: obj})
        locked = lock_head_shape_keys(instance, {"head_lod0__jaw_open"})
        blocks = obj.data.shape_keys.key_blocks
        assert locked == 1  # only brow_down is locked
        assert blocks[SHAPE_KEY_BASIS_NAME].lock_shape is False
        assert blocks["head_lod0__jaw_open"].lock_shape is False
        assert blocks["head_lod0__brow_down"].lock_shape is True
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_lock_all_except_basis_locks_every_target() -> None:
    obj = _make_quad("lock_all", ["head_lod0__jaw_open", "head_lod0__brow_down"])
    try:
        instance = _FakeInstance("Ada", {0: obj})
        # Pre-unlock everything to prove the helper re-locks.
        for block in obj.data.shape_keys.key_blocks:
            block.lock_shape = False
        locked = lock_all_head_shape_keys_except_basis(instance)
        blocks = obj.data.shape_keys.key_blocks
        assert locked == 2
        assert blocks[SHAPE_KEY_BASIS_NAME].lock_shape is False
        assert blocks["head_lod0__jaw_open"].lock_shape is True
        assert blocks["head_lod0__brow_down"].lock_shape is True
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_lock_helpers_no_shape_keys_is_noop() -> None:
    mesh = bpy.data.meshes.new("plain")
    mesh.from_pydata(_BASE_POSITIONS, [], [_QUAD_FACE])
    mesh.update()
    obj = bpy.data.objects.new("plain", mesh)
    bpy.context.scene.collection.objects.link(obj)
    try:
        instance = _FakeInstance("Ada", {0: obj})
        assert lock_all_head_shape_keys_except_basis(instance) == 0
        assert mesh.shape_keys is None
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
