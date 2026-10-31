"""Tests for the Mesh Editor's data transfer / Topology Shim core.

These run inside Blender's Python (so ``bpy`` / ``bmesh`` are available). They
cover the pure-numpy barycentric interpolation, identical-topology detection and
direct mapping, the world-space nearest-point transfer for differing topology,
shape-key delta transfer, the ``delete_existing`` clearing, armature binding,
and the exported ``.npz`` shim round-trip."""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
import pytest

from character_dna.editors.mesh_editor import core


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def created() -> list[bpy.types.Object]:
    """Track objects created during a test and remove them on teardown."""
    objects: list[bpy.types.Object] = []
    yield objects
    for obj in objects:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(data, bpy.types.Mesh) and data.users == 0:
            bpy.data.meshes.remove(data)


def _make_mesh(
    created: list[bpy.types.Object],
    name: str,
    verts: list[tuple[float, float, float]],
    faces: list[list[int]],
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    created.append(obj)
    return obj


_QUAD_VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
_QUAD_FACE = [[0, 1, 2, 3]]


def _set_group(obj: bpy.types.Object, name: str, weights: list[float]) -> None:
    vertex_group = obj.vertex_groups.new(name=name)
    for index, weight in enumerate(weights):
        if weight:
            vertex_group.add((index,), weight, "REPLACE")


def _read_group(obj: bpy.types.Object, name: str) -> list[float]:
    vertex_group = obj.vertex_groups[name]
    out: list[float] = []
    for vertex in obj.data.vertices:
        try:
            out.append(vertex_group.weight(vertex.index))
        except RuntimeError:
            out.append(0.0)
    return out


# ---------------------------------------------------------------------------
# Pure-numpy interpolation
# ---------------------------------------------------------------------------
def test_direct_correspondence_is_identity() -> None:
    correspondence = core._direct_correspondence(4, "3D")
    assert correspondence.tri.shape == (4, 3)
    assert np.allclose(correspondence.bary[:, 0], 1.0)
    values = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    assert np.array_equal(core._interpolate_scalar(values, correspondence), values)


def test_interpolate_scalar_centroid() -> None:
    correspondence = core.Correspondence(
        tri=np.array([[0, 1, 2]], dtype=np.int32),
        bary=np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32),
        missed=np.array([False]),
        source_vertex_count=3,
        target_vertex_count=1,
        space="3D",
    )
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert core._interpolate_scalar(values, correspondence)[0] == pytest.approx(2.0)


def test_interpolate_scalar_missed_falls_back_to_nearest() -> None:
    correspondence = core.Correspondence(
        tri=np.array([[1, 0, 2]], dtype=np.int32),
        bary=np.array([[0.5, 0.25, 0.25]], dtype=np.float32),
        missed=np.array([True]),
        source_vertex_count=3,
        target_vertex_count=1,
        space="3D",
    )
    values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    assert core._interpolate_scalar(values, correspondence)[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Topology detection + direct mapping
# ---------------------------------------------------------------------------
def test_identical_topology_detection(created: list[bpy.types.Object]) -> None:
    source = _make_mesh(created, "src_quad", _QUAD_VERTS, _QUAD_FACE)
    same = _make_mesh(created, "same_quad", [(x, y, z + 5) for x, y, z in _QUAD_VERTS], _QUAD_FACE)
    triangle = _make_mesh(created, "tri", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [[0, 1, 2]])
    assert core._identical_topology(source, same) is True
    assert core._identical_topology(source, triangle) is False


def test_transfer_weights_identical_topology_round_trips(created: list[bpy.types.Object]) -> None:
    source = _make_mesh(created, "src_quad", _QUAD_VERTS, _QUAD_FACE)
    target = _make_mesh(created, "tgt_quad", _QUAD_VERTS, _QUAD_FACE)
    _set_group(source, "FACIAL_C_Jaw", [0.0, 1.0, 1.0, 0.0])

    correspondence = core.build_correspondence(source, target, "3D")
    written = core.transfer_weights(correspondence, source, target, {"FACIAL_C_Jaw"})

    assert written == 1
    assert _read_group(target, "FACIAL_C_Jaw") == pytest.approx([0.0, 1.0, 1.0, 0.0])


def test_transfer_weights_3d_barycentric(created: list[bpy.types.Object]) -> None:
    source = _make_mesh(created, "src_quad", _QUAD_VERTS, _QUAD_FACE)
    _set_group(source, "FACIAL_C_Jaw", [0.0, 1.0, 1.0, 0.0])  # weight == corner x
    target = _make_mesh(created, "tgt_tri", [(0.2, 0.5, 0.1), (0.8, 0.5, 0.1), (0.5, 0.5, 0.2)], [[0, 1, 2]])

    correspondence = core.build_correspondence(source, target, "3D")
    core.transfer_weights(correspondence, source, target, {"FACIAL_C_Jaw"})

    assert _read_group(target, "FACIAL_C_Jaw") == pytest.approx([0.2, 0.8, 0.5], abs=1e-4)


def test_transfer_weights_delete_existing_clears_groups(created: list[bpy.types.Object]) -> None:
    source = _make_mesh(created, "src_quad", _QUAD_VERTS, _QUAD_FACE)
    target = _make_mesh(created, "tgt_quad", _QUAD_VERTS, _QUAD_FACE)
    _set_group(source, "FACIAL_C_Jaw", [0.0, 1.0, 1.0, 0.0])
    _set_group(target, "junk", [1.0, 1.0, 1.0, 1.0])

    correspondence = core.build_correspondence(source, target, "3D")
    core.transfer_weights(correspondence, source, target, {"FACIAL_C_Jaw"}, delete_existing=True)

    names = [vertex_group.name for vertex_group in target.vertex_groups]
    assert "junk" not in names
    assert "FACIAL_C_Jaw" in names


# ---------------------------------------------------------------------------
# Shape-key transfer
# ---------------------------------------------------------------------------
def test_transfer_shape_keys_identical_topology(created: list[bpy.types.Object]) -> None:
    source = _make_mesh(created, "src_quad", _QUAD_VERTS, _QUAD_FACE)
    target = _make_mesh(created, "tgt_quad", _QUAD_VERTS, _QUAD_FACE)
    source.shape_key_add(name="Basis", from_mix=False)
    key_block = source.shape_key_add(name="jawOpen", from_mix=False)
    key_block.data[1].co.z += 0.5
    key_block.slider_min = -1.0
    key_block.slider_max = 2.0

    correspondence = core.build_correspondence(source, target, "3D")
    written = core.transfer_shape_keys(correspondence, source, target)

    assert written == 1
    target_key = target.data.shape_keys.key_blocks["jawOpen"]
    assert target_key.data[1].co.z == pytest.approx(0.5)
    assert target_key.slider_min == pytest.approx(-1.0)
    assert target_key.slider_max == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Armature binding
# ---------------------------------------------------------------------------
def test_ensure_armature_modifier_adds_then_skips(created: list[bpy.types.Object]) -> None:
    target = _make_mesh(created, "tgt_quad", _QUAD_VERTS, _QUAD_FACE)
    armature = bpy.data.objects.new("rig", bpy.data.armatures.new("rig"))
    bpy.context.scene.collection.objects.link(armature)
    created.append(armature)

    assert core.ensure_armature_modifier(target, armature) is True
    assert any(m.type == "ARMATURE" and m.object == armature for m in target.modifiers)
    assert core.ensure_armature_modifier(target, armature) is False
    assert sum(1 for m in target.modifiers if m.type == "ARMATURE") == 1


# ---------------------------------------------------------------------------
# Shim persistence (exported .npz file)
# ---------------------------------------------------------------------------
def test_shim_file_round_trip(created: list[bpy.types.Object], tmp_path: Path) -> None:
    source = _make_mesh(created, "src_quad", _QUAD_VERTS, _QUAD_FACE)
    target = _make_mesh(created, "tgt_tri", [(0.2, 0.5, 0.1), (0.8, 0.5, 0.1), (0.5, 0.5, 0.2)], [[0, 1, 2]])

    correspondence = core.build_correspondence(source, target, "3D")
    path = core.save_shim_to_file(correspondence, tmp_path, source, target)
    assert path.is_file()
    assert path in core.list_shim_files(tmp_path)

    loaded = core.load_shim_from_file(path)
    assert loaded is not None
    loaded_correspondence, meta = loaded
    assert meta["source_object"] == source.name
    assert meta["target_object"] == target.name
    assert loaded_correspondence.matches(source, target)
    assert np.array_equal(loaded_correspondence.tri, correspondence.tri)
    assert np.allclose(loaded_correspondence.bary, correspondence.bary, atol=1e-6)
