"""Tests for the "Transfer Shape From Selected Mesh" Raw Control Editor
operator and its supporting helpers.

Two layers are covered, both running against the real ``bpy`` module
provided by the test environment:

* The centralized UV-space barycentric solver
  (:func:`character_dna.editors.shared.utilities.calculate_dna_mesh_vertex_positions`)
  -- now shared between LOD propagation and this operator. A pair of
  unit quads with identical UV layouts but different 3D shapes proves
  the solver copies the donor's shape onto the target exactly.

* The donor-resolution gate
  (:func:`...raw_control_editor.utilities.resolve_donor_mesh` /
  :func:`selected_donor_mesh_candidates`) which must exclude rig and
  target meshes and require exactly one (or the active) eligible mesh.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import bpy
import numpy as np
import pytest

from character_dna.editors.raw_control_editor.utilities import (
    RawControlEditorError,
    resolve_donor_mesh,
    selected_donor_mesh_candidates,
)
from character_dna.editors.shared.utilities import calculate_dna_mesh_vertex_positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A single quad: corner verts in CCW order with UVs pinned to the unit
# square so two quads share an identical UV layout regardless of their
# 3D shape.
_QUAD_FACE = [0, 1, 2, 3]
_QUAD_UVS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _make_quad(name: str, positions: list[tuple[float, float, float]], *, uvs: bool = True) -> bpy.types.Object:
    """Create and scene-link a single-quad mesh object with the given
    corner ``positions`` and (optionally) the canonical unit-square UVs."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(positions, [], [_QUAD_FACE])
    mesh.update()
    if uvs:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for loop_index, loop in enumerate(mesh.loops):
            uv_layer.data[loop_index].uv = _QUAD_UVS[loop.vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _read_positions(obj: bpy.types.Object) -> np.ndarray:
    mesh = obj.data
    count = len(mesh.vertices)
    flat = np.zeros(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    return flat.reshape(count, 3)


class _Row:
    """Minimal stand-in for a ``TargetMeshItem`` row."""

    def __init__(self, scene_object: Any) -> None:
        self.scene_object = scene_object


class _FakeInstance:
    """Duck-typed ``RigInstance`` exposing only what the donor helpers read."""

    def __init__(self, head: list[Any], body: list[Any], head_mesh: Any = None) -> None:
        self.output = SimpleNamespace(head_item_list=head, body_item_list=body)
        self.edit_target = SimpleNamespace(mesh=head_mesh)


class _FakeEditor:
    """Duck-typed ``RawControlEditorProperties`` for donor resolution."""

    def __init__(self, target_rows: list[_Row], active_index: int = 0) -> None:
        self.target_meshes = target_rows
        self.target_meshes_active_index = active_index


@pytest.fixture(autouse=True)
def _clean_scene() -> Any:
    """Remove any objects/meshes created during a test so object names
    and the selection state never bleed across tests."""
    before_objects = set(bpy.data.objects.keys())
    before_meshes = set(bpy.data.meshes.keys())
    yield
    for name in set(bpy.data.objects.keys()) - before_objects:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
    for name in set(bpy.data.meshes.keys()) - before_meshes:
        mesh = bpy.data.meshes.get(name)
        if mesh is not None:
            bpy.data.meshes.remove(mesh, do_unlink=True)


def _select_only(*objects: bpy.types.Object, active: bpy.types.Object | None = None) -> None:
    """Deselect everything, then select ``objects`` and set ``active``."""
    # Direct data API creation/removal leaves view-layer bases stale until updated.
    bpy.context.view_layer.update()
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active if active is not None else (objects[0] if objects else None)


# ---------------------------------------------------------------------------
# Centralized UV-barycentric solver
# ---------------------------------------------------------------------------


def test_transfer_copies_donor_shape_onto_target() -> None:
    """With identical UV layouts each target corner maps exactly onto the
    matching donor corner, so the donor's 3D shape is copied verbatim."""
    donor_shape = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 1.0, 1.0), (0.0, 1.0, 0.25)]
    target_shape = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    donor = _make_quad("donor_quad", donor_shape)
    target = _make_quad("target_quad", target_shape)

    result = calculate_dna_mesh_vertex_positions({"name": donor.name}, {"name": target.name})

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 3)
    np.testing.assert_allclose(result, np.array(donor_shape, dtype=np.float32), atol=1e-5)


def test_transfer_respects_target_vertex_count() -> None:
    """The result always has the target's vertex count, even when the
    donor has a different topology (here a denser grid)."""
    donor = _make_quad("donor_quad", [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])
    target = _make_quad("target_quad", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)])

    result = calculate_dna_mesh_vertex_positions({"name": donor.name}, {"name": target.name})

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(target.data.vertices)


def test_transfer_returns_empty_without_uv_layer() -> None:
    """A mesh with no active UV layer cannot be matched, so the solver
    returns an empty dict rather than raising."""
    donor = _make_quad("donor_quad", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
    target = _make_quad("target_quad", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)], uvs=False)

    result = calculate_dna_mesh_vertex_positions({"name": donor.name}, {"name": target.name})

    assert result == {}


# ---------------------------------------------------------------------------
# Donor resolution gate
# ---------------------------------------------------------------------------


def _flat_quad(name: str) -> bpy.types.Object:
    return _make_quad(name, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)])


def test_resolve_donor_requires_a_selection() -> None:
    instance = _FakeInstance(head=[], body=[])
    editor = _FakeEditor(target_rows=[])
    _select_only()  # nothing selected

    with pytest.raises(RawControlEditorError, match="Select one donor mesh"):
        resolve_donor_mesh(instance, editor)


def test_resolve_donor_returns_single_eligible_mesh() -> None:
    donor = _flat_quad("donor_quad")
    instance = _FakeInstance(head=[], body=[])
    editor = _FakeEditor(target_rows=[])
    _select_only(donor, active=donor)

    assert resolve_donor_mesh(instance, editor) is donor


def test_resolve_donor_excludes_rig_and_target_meshes() -> None:
    rig_mesh = _flat_quad("rig_quad")
    target_mesh = _flat_quad("target_quad")
    instance = _FakeInstance(head=[_Row(rig_mesh)], body=[], head_mesh=None)
    editor = _FakeEditor(target_rows=[_Row(target_mesh)])
    _select_only(rig_mesh, target_mesh, active=rig_mesh)

    # Both selected meshes are excluded, so there is no eligible donor.
    assert selected_donor_mesh_candidates(instance, editor) == []
    with pytest.raises(RawControlEditorError, match="Select one donor mesh"):
        resolve_donor_mesh(instance, editor)


def test_resolve_donor_prefers_active_among_several() -> None:
    first = _flat_quad("donor_a")
    second = _flat_quad("donor_b")
    instance = _FakeInstance(head=[], body=[])
    editor = _FakeEditor(target_rows=[])
    _select_only(first, second, active=second)

    assert resolve_donor_mesh(instance, editor) is second


def test_resolve_donor_errors_when_several_and_none_active() -> None:
    first = _flat_quad("donor_a")
    second = _flat_quad("donor_b")
    other = _flat_quad("other")  # eligible, but kept as the active object
    instance = _FakeInstance(head=[_Row(other)], body=[])
    editor = _FakeEditor(target_rows=[])
    # ``other`` is an excluded rig mesh and is active; two eligible meshes
    # remain selected with no eligible active object -> ambiguous.
    _select_only(first, second, other, active=other)

    with pytest.raises(RawControlEditorError, match="select only one"):
        resolve_donor_mesh(instance, editor)


def test_resolve_donor_allows_deformed_head_mesh() -> None:
    """The rig's live head mesh is re-allowed as a donor so its deformed
    shape can be copied onto the target, even though it is also listed as
    the ``head_lod0_mesh`` output item."""
    head_mesh = _flat_quad("head_lod0_mesh")
    target_mesh = _flat_quad("target_quad")
    # The head mesh is registered both as ``head_mesh`` and as its output item.
    instance = _FakeInstance(head=[_Row(head_mesh)], body=[], head_mesh=head_mesh)
    editor = _FakeEditor(target_rows=[_Row(target_mesh)])
    _select_only(head_mesh, active=head_mesh)

    assert selected_donor_mesh_candidates(instance, editor) == [head_mesh]
    assert resolve_donor_mesh(instance, editor) is head_mesh


def test_head_mesh_excluded_when_it_is_also_the_target() -> None:
    """If the head mesh is itself the active target row it stays excluded
    -- a mesh cannot be its own donor."""
    head_mesh = _flat_quad("head_lod0_mesh")
    instance = _FakeInstance(head=[_Row(head_mesh)], body=[], head_mesh=head_mesh)
    editor = _FakeEditor(target_rows=[_Row(head_mesh)])
    _select_only(head_mesh, active=head_mesh)

    assert selected_donor_mesh_candidates(instance, editor) == []
    with pytest.raises(RawControlEditorError, match="Select one donor mesh"):
        resolve_donor_mesh(instance, editor)
