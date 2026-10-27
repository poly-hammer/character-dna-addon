"""Regression guard for the blend-shape-target delta writeback format.

The OpenRigLogic ``GeometryWriter.setBlendShapeTargetDeltas`` typemap requires the
``deltas`` argument to be a list of ``[x, y, z]`` **lists** of plain Python floats
(the same shape ``setVertexPositions`` accepts). Passing a list of **tuples** (or
``mathutils`` scalars) fails hard with::

    SystemError: returned a result with an exception set
    TypeError: bad argument type for built-in operation

Two production sites build these deltas and both must follow the contract:

* :func:`character_dna.editors.shape_key_editor.core.compute_dna_deltas`
  (Shape Key Editor commit), and
* ``character_dna.dna_io.calibrator.DNACalibrator.calibrate_shape_keys``
  (Export Selected Component with the *calibrate* method).

This module pins the binding contract directly (so a binding upgrade that changes
the requirement is caught) and verifies the editor helper emits the accepted shape.
"""

import pytest

from character_dna.dna_io import get_dna_reader, get_dna_writer
from constants import HEAD_DNA_FILE


# Shape-key editing lives in the Pro ``editors`` submodule; skip cleanly on a Free
# checkout where it is absent.
core = pytest.importorskip("character_dna.editors.shape_key_editor.core")


def _first_mesh_with_blend_shapes(reader) -> int | None:
    for mesh_index in range(reader.getMeshCount()):
        if reader.getBlendShapeTargetCount(mesh_index) > 0:
            return mesh_index
    return None


def test_writer_accepts_list_of_lists_and_round_trips(tmp_path):
    """A list of ``[x, y, z]`` float lists must be accepted and written verbatim."""
    reader = get_dna_reader(HEAD_DNA_FILE)
    assert reader is not None

    mesh_index = _first_mesh_with_blend_shapes(reader)
    if mesh_index is None:
        pytest.skip("Test DNA has no blend shape targets.")

    vertex_indices = [0, 1, 2]
    deltas = [[0.1, 0.2, 0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, -0.9]]

    output_path = tmp_path / "head_blend_shape_roundtrip.dna"
    writer = get_dna_writer(output_path)
    from character_dna.bindings import dna  # type: ignore[reportAttributeAccessIssue]

    writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    writer.setBlendShapeTargetVertexIndices(meshIndex=mesh_index, blendShapeTargetIndex=0, vertexIndices=vertex_indices)
    writer.setBlendShapeTargetDeltas(meshIndex=mesh_index, blendShapeTargetIndex=0, deltas=deltas)
    writer.write()
    assert dna.Status.isOk()

    result = get_dna_reader(output_path)
    assert result is not None
    xs = list(result.getBlendShapeTargetDeltaXs(mesh_index, 0))
    ys = list(result.getBlendShapeTargetDeltaYs(mesh_index, 0))
    zs = list(result.getBlendShapeTargetDeltaZs(mesh_index, 0))
    assert xs == pytest.approx([d[0] for d in deltas], abs=1e-5)
    assert ys == pytest.approx([d[1] for d in deltas], abs=1e-5)
    assert zs == pytest.approx([d[2] for d in deltas], abs=1e-5)


def test_writer_rejects_list_of_tuples(tmp_path):
    """Document/lock the contract: tuples are rejected by the SWIG typemap. This is
    the exact failure the production fix avoids by emitting lists."""
    reader = get_dna_reader(HEAD_DNA_FILE)
    assert reader is not None
    mesh_index = _first_mesh_with_blend_shapes(reader)
    if mesh_index is None:
        pytest.skip("Test DNA has no blend shape targets.")

    output_path = tmp_path / "head_blend_shape_tuples.dna"
    writer = get_dna_writer(output_path)
    from character_dna.bindings import dna  # type: ignore[reportAttributeAccessIssue]

    writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    writer.setBlendShapeTargetVertexIndices(meshIndex=mesh_index, blendShapeTargetIndex=0, vertexIndices=[0, 1])

    with pytest.raises((TypeError, SystemError)):
        writer.setBlendShapeTargetDeltas(
            meshIndex=mesh_index, blendShapeTargetIndex=0, deltas=[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]
        )


def test_compute_dna_deltas_emits_list_of_float_lists():
    """The editor helper must return ``list[list[float]]`` of plain Python floats so
    the deltas are accepted by ``setBlendShapeTargetDeltas``."""
    import bpy

    from character_dna.constants import SHAPE_KEY_BASIS_NAME

    mesh = bpy.data.meshes.new("regression_shape_key_mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)], [], [(0, 1, 3, 2)])
    mesh.update()
    mesh_object = bpy.data.objects.new("regression_shape_key_object", mesh)
    bpy.context.scene.collection.objects.link(mesh_object)
    try:
        mesh_object.shape_key_add(name=SHAPE_KEY_BASIS_NAME, from_mix=False)
        key_block = mesh_object.shape_key_add(name="head_mesh__regression", from_mix=False)
        # Move two vertices well beyond the delta threshold.
        key_block.data[0].co.x += 0.5
        key_block.data[1].co.z -= 0.25

        vertex_indices, deltas = core.compute_dna_deltas(mesh_object, key_block, linear_modifier=1.0)

        assert isinstance(deltas, list) and len(deltas) >= 2
        assert len(vertex_indices) == len(deltas)
        for delta in deltas:
            assert type(delta) is list
            assert len(delta) == 3
            assert all(type(component) is float for component in delta)
    finally:
        bpy.data.objects.remove(mesh_object, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def test_sparse_native_write_preserves_untouched_fields(tmp_path):
    from character_dna.dna_io import merged_graph
    from character_dna.dna_io.coordinates import write_shape_targets
    from test_native_merged_graph import first_difference

    path = tmp_path / "head.dna"
    path.write_bytes(HEAD_DNA_FILE.read_bytes())
    expected = merged_graph.from_bytes(path.read_bytes())
    mesh = next(index for index, item in enumerate(expected["geom1.1"]["meshes"]) if item["blendShapeTargets"])
    indices = [0, 1, 2]
    deltas = [[0.25, -0.5, 0.75], [-1.0, 2.0, -3.0], [0.0, 0.0, 0.0]]
    target = expected["geom1.1"]["meshes"][mesh]["blendShapeTargets"][0]
    target["vertexIndices"] = indices
    target["deltas"] = {"xs": [0.25, -1.0, 0.0], "ys": [0.75, -3.0, 0.0], "zs": [0.5, -2.0, 0.0]}
    write_shape_targets(path, {(mesh, 0): (indices, deltas)})
    assert first_difference(merged_graph.from_bytes(path.read_bytes()), expected) is None

    before = path.read_bytes()
    with pytest.raises(ValueError, match="target-only"):
        write_shape_targets(path, {(mesh, 65535): ([0], [[0.0, 0.0, 0.0]])})
    assert path.read_bytes() == before
