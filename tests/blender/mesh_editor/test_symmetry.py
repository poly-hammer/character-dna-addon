"""Tests for the shared mesh-symmetry primitives the Mesh Editor uses for
UV-based mirror / flip cleanup."""

from __future__ import annotations

import bpy
import pytest

from character_dna.editors.shared import symmetry


VertexSide = symmetry.VertexSide
MirrorDirection = symmetry.MirrorDirection


# ---------------------------------------------------------------------------
# UV pairing
# ---------------------------------------------------------------------------
def test_build_uv_vertex_pairs_classifies_and_pairs() -> None:
    # v0 left (u>0.5), v1 its right mirror, v2 center (u==0.5), v3 left orphan.
    uv = [(0.75, 0.25), (0.25, 0.25), (0.5, 0.5), (0.9, 0.8)]
    partner, side = symmetry.build_uv_vertex_pairs(uv)
    assert side[0] == VertexSide.LEFT
    assert side[1] == VertexSide.RIGHT
    assert side[2] == VertexSide.CENTER
    assert side[3] == VertexSide.LEFT
    assert partner[0] == 1
    assert partner[1] == 0
    assert partner[2] == 2
    assert partner[3] == 3  # orphan -> self


# ---------------------------------------------------------------------------
# Absolute mirror / flip
# ---------------------------------------------------------------------------
_POSITIONS = [(1.0, 2.0, 3.0), (-9.0, 9.0, 9.0), (0.0, 7.0, 8.0)]
_PARTNER = [1, 0, 2]
_SIDE = [VertexSide.LEFT, VertexSide.RIGHT, VertexSide.CENTER]


def test_mirror_positions_copies_source_onto_destination() -> None:
    out = symmetry.mirror_positions(_POSITIONS, _PARTNER, _SIDE, MirrorDirection.LEFT_TO_RIGHT)
    # destination RIGHT (index 1) takes the X-mirror of its LEFT partner (index 0).
    assert out[1] == (-1.0, 2.0, 3.0)
    assert out[0] == (1.0, 2.0, 3.0)  # source side untouched
    assert out[2] == (0.0, 7.0, 8.0)  # center untouched (snap_center default False)


def test_mirror_positions_snap_center() -> None:
    positions = [(1.0, 2.0, 3.0), (-9.0, 9.0, 9.0), (0.3, 7.0, 8.0)]
    out = symmetry.mirror_positions(positions, _PARTNER, _SIDE, MirrorDirection.LEFT_TO_RIGHT, snap_center=True)
    assert out[2] == (0.0, 7.0, 8.0)


def test_flip_positions_swaps_pairs() -> None:
    out = symmetry.flip_positions(_POSITIONS, _PARTNER, _SIDE)
    assert out[0] == (9.0, 9.0, 9.0)  # mirror of partner (index 1)
    assert out[1] == (-1.0, 2.0, 3.0)  # mirror of partner (index 0)
    assert out[2] == (0.0, 7.0, 8.0)  # center reflected in place


# ---------------------------------------------------------------------------
# Per-vertex UV reader
# ---------------------------------------------------------------------------
def test_get_per_vertex_uv() -> None:
    mesh = bpy.data.meshes.new("uv_quad")
    mesh.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)], [], [[0, 1, 2, 3]])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    corner_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for loop_index, loop in enumerate(mesh.loops):
        uv_layer.data[loop_index].uv = corner_uvs[loop.vertex_index]
    obj = bpy.data.objects.new("uv_quad", mesh)
    bpy.context.scene.collection.objects.link(obj)
    try:
        result = symmetry.get_per_vertex_uv(obj)
        assert result is not None
        assert len(result) == 4
        assert tuple(result[1]) == pytest.approx((1.0, 0.0))
        assert tuple(result[3]) == pytest.approx((0.0, 1.0))
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
