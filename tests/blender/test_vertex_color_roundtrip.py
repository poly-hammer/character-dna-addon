"""Real DNA layouts and Blender corner colors survive custom mesh round trips."""

import json

from types import SimpleNamespace

import bmesh
import bpy
import numpy as np
import pytest

from character_dna.dna_io import DNAExporter, DNAImporter, get_dna_reader, get_dna_writer, release_dna_handle
from constants import TEST_DNA_FOLDER


def write_surface(path, *, lods=(0,), lod_count=8, component="head"):
    writer = get_dna_writer(path)
    try:
        writer.setName("Template")
        writer.setLODCount(lod_count)
        writer.setMeshName(0, f"{component}_lod0_mesh")
        writer.setMeshIndices(0, [0])
        writer.setMeshIndices(1, [])
        for lod in range(lod_count):
            writer.setLODMeshMapping(lod, 0 if lod in lods else 1)
        writer.setVertexPositions(0, [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]])
        writer.setVertexNormals(0, [[0, 0, 1]])
        writer.setVertexTextureCoordinates(0, [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [1, 1]])
        # Layout IDs deliberately differ from position IDs, including UV seams.
        writer.setVertexLayouts(0, [[p, i, 0] for i, p in enumerate([2, 0, 3, 1, 0, 2])])
        writer.setFaceVertexLayoutIndices(0, 0, [1, 3, 0])
        writer.setFaceVertexLayoutIndices(0, 1, [4, 5, 2])
        writer.write()
    finally:
        release_dna_handle(writer)
    return path


def import_surface(path, *, component="head", name="Renamed"):
    properties = SimpleNamespace(**{f"import_lod{i}": True for i in range(8)})
    instance = SimpleNamespace(name=name, head_dna_file_path=str(path), body_dna_file_path=str(path))
    importer = DNAImporter(instance, properties, 0.01, component_type=component)
    bm = bmesh.new()
    importer.set_mesh_vertex_positions(0, bm)
    importer.set_mesh_face_layout(0, bm)
    importer.set_vertex_colors(0, bm)
    return importer, bm


@pytest.mark.parametrize("data_type", ["BYTE_COLOR", "FLOAT_COLOR"])
@pytest.mark.parametrize("uv_seam", [False, True])
def test_export_import_corner_colors(tmp_path, data_type, uv_seam):
    """Includes color seams without UV seams, alpha, and stale BMLoop indices."""
    source = write_surface(tmp_path / "source.dna")
    reader = get_dna_reader(source)
    mesh = bpy.data.meshes.new("color_roundtrip")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2), (0, 2, 3)])
    uv = mesh.uv_layers.new()
    for index, value in enumerate([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)]):
        uv.data[index].uv = (value[0] + (2 if uv_seam and index >= 3 else 0), value[1])
    attr = mesh.color_attributes.new(name="Color", type=data_type, domain="CORNER")
    colors = [(0.1, 0.2, 0.3, 0.4), (1, 0, 0, 1), (0, 1, 0, 1), (0.8, 0.3, 0.2, 0.7), (0, 0, 1, 1), (0, 1, 1, 0)]
    for datum, color in zip(attr.data, colors, strict=True):
        datum.color = color
    expected = [tuple(datum.color) for datum in attr.data]
    obj = bpy.data.objects.new("Original_head_lod0_mesh", mesh)
    instance = SimpleNamespace(
        name="Original",
        editor_component="head",
        head_mesh=obj,
        head_rig=None,
        head_dna_file_path=str(source),
        output=SimpleNamespace(folder_path=str(tmp_path), format="binary"),
    )
    exporter = DNAExporter(instance, 100, file_name="head.dna", reader=reader, vertex_groups=False)
    exporter._vertex_color_data = [{"indices": [], "values": []}]
    importer = None
    imported_bm = None
    imported_mesh = None
    try:
        exporter._export_mesh(obj, 0)
        exporter._dna_writer.write()
        exporter.save_vertex_colors()
        importer, imported_bm = import_surface(tmp_path / "head.dna")
        imported_mesh = bpy.data.meshes.new("imported_colors")
        imported_bm.to_mesh(imported_mesh)
        result = imported_mesh.color_attributes["Color"]
        assert result.domain == "CORNER"
        assert result.data_type == data_type
        np.testing.assert_allclose([tuple(datum.color) for datum in result.data], expected, atol=1e-7)
        assert not importer._default_vertex_color_layout
        assert importer._get_lod_settings() == [(0, True)]
    finally:
        if imported_bm is not None:
            imported_bm.free()
        if importer is not None:
            release_dna_handle(importer._dna_reader)
        release_dna_handle(exporter._dna_writer)
        release_dna_handle(reader)
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
        if imported_mesh is not None:
            bpy.data.meshes.remove(imported_mesh)


@pytest.mark.parametrize("filename", ["Original_vertex_colors.json", "Original_head_vertex_colors.json"])
def test_legacy_export_uses_layout_ids_and_manifest(tmp_path, filename):
    path = write_surface(tmp_path / "head.dna")
    (tmp_path / "ExportManifest.json").write_text(json.dumps({"metaHumanName": "Original"}))
    palette = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0.5], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]
    (tmp_path / filename).write_text(json.dumps([{"indices": list(range(6)), "values": palette}]))
    importer, bm = import_surface(path)
    try:
        layer = bm.loops.layers.color["Color"]
        for face in bm.faces:
            layouts = importer._dna_reader.getFaceVertexLayoutIndices(0, face.index)
            for loop, index in zip(face.loops, layouts, strict=True):
                np.testing.assert_allclose(tuple(loop[layer]), palette[index], atol=1 / 255)
        assert not importer._default_vertex_color_layout
    finally:
        bm.free()
        release_dna_handle(importer._dna_reader)


def test_custom_topology_without_sidecar_does_not_get_template_colors(tmp_path):
    importer, bm = import_surface(write_surface(tmp_path / "head.dna"))
    try:
        assert not bm.loops.layers.color.keys()
    finally:
        bm.free()
        release_dna_handle(importer._dna_reader)


def test_standard_metahuman_keeps_bundled_position_colors():
    importer, bm = import_surface(TEST_DNA_FOLDER / "ada" / "head.dna")
    try:
        assert importer._default_vertex_color_layout
        indices, values = importer.get_dna_vertex_colors(0)
        positions = importer._dna_reader.getVertexLayoutPositionIndices(0)
        layer = bm.loops.layers.color["Color"]
        expected = []
        actual = []
        for face in bm.faces:
            layouts = importer._dna_reader.getFaceVertexLayoutIndices(0, face.index)
            for loop, layout in zip(face.loops, layouts, strict=True):
                actual.append(tuple(loop[layer]))
                expected.append(values[indices[positions[layout]]])
        np.testing.assert_allclose(actual, expected, atol=1 / 255)
    finally:
        bm.free()
        release_dna_handle(importer._dna_reader)


def test_head_body_sidecars_do_not_overwrite_each_other(tmp_path):
    for component, color in [("head", [0, 1, 0, 1]), ("body", [0, 0, 1, 1])]:
        path = write_surface(tmp_path / f"{component}.dna", component=component)
        exporter = DNAExporter.__new__(DNAExporter)
        exporter._include_vertex_colors = True
        exporter._target_dna_file = path
        exporter._prefix = "Original"
        exporter._vertex_color_data = [{"indices": [0] * 6, "values": [color]}]
        exporter.save_vertex_colors()
    for component, color in [("head", [0, 1, 0, 1]), ("body", [0, 0, 1, 1])]:
        importer, bm = import_surface(tmp_path / f"{component}.dna", component=component)
        try:
            layer = bm.loops.layers.color["Color"]
            assert all(tuple(loop[layer]) == tuple(color) for face in bm.faces for loop in face.loops)
        finally:
            bm.free()
            release_dna_handle(importer._dna_reader)


def test_invalid_color_sidecar_is_not_partially_applied(tmp_path):
    path = write_surface(tmp_path / "head.dna")
    (tmp_path / "head_vertex_colors.json").write_text(
        json.dumps([{"indices": [0, 0, 9, 0, 0, 0], "values": [[1, 0, 0, 1]]}])
    )
    importer, bm = import_surface(path)
    try:
        assert not bm.loops.layers.color.keys()
    finally:
        bm.free()
        release_dna_handle(importer._dna_reader)
