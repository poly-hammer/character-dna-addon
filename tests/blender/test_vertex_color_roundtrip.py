"""Real DNA layouts and Blender corner colors survive custom mesh round trips."""

import json

from types import SimpleNamespace

import bmesh
import bpy
import numpy as np
import pytest

from character_dna.bindings import dna
from character_dna.dna_io import DNAExporter, DNAImporter, get_dna_reader, get_dna_writer, mesh_data, release_dna_handle
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
        output=SimpleNamespace(folder_path=str(tmp_path), format="binary", head_topology_type="Custom"),
        get=lambda *_: None,
    )
    exporter = DNAExporter(instance, 100, file_name="head.dna", reader=reader, vertex_groups=False)
    importer = None
    imported = None
    imported_mesh = None
    try:
        exporter._export_mesh(obj, 0)
        exporter._export_component_extensions()
        exporter._dna_writer.write()
        assert not list(tmp_path.glob("*_vertex_colors.json"))
        path = tmp_path / "head.dna"
        options = SimpleNamespace(
            import_vertex_colors=True,
            import_vertex_groups=False,
            import_normals=False,
            **{f"import_lod{i}": True for i in range(8)},
        )
        importer = DNAImporter(SimpleNamespace(name="Imported", head_dna_file_path=str(path)), options, 0.01)
        importer.initialize_dna_data()
        imported = importer.create_mesh_object(0, "head_lod0_mesh")
        imported_mesh = imported.data
        result = imported_mesh.color_attributes["Color"]
        assert result.domain == "CORNER"
        assert result.data_type == data_type
        np.testing.assert_allclose([tuple(datum.color) for datum in result.data], expected, atol=1e-7)
        assert importer._custom_topology
        assert importer._get_lod_settings() == [(0, True)]
    finally:
        if imported is not None:
            bpy.data.objects.remove(imported, do_unlink=True)
        if importer is not None:
            release_dna_handle(importer._dna_reader)
        release_dna_handle(exporter._dna_writer)
        release_dna_handle(reader)
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
        if imported_mesh is not None:
            bpy.data.meshes.remove(imported_mesh)


@pytest.mark.parametrize(
    "filename", ["head_vertex_colors.json", "Original_vertex_colors.json", "Original_head_vertex_colors.json"]
)
def test_sidecars_are_ignored(tmp_path, filename):
    path = write_surface(tmp_path / "head.dna")
    (tmp_path / "ExportManifest.json").write_text(json.dumps({"metaHumanName": "Original"}))
    (tmp_path / filename).write_text("invalid JSON must never be read")
    importer, bm = import_surface(path)
    try:
        assert not bm.loops.layers.color.keys()
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


@pytest.mark.parametrize("topology_type", ["", "MetaHuman"])
def test_standard_metahuman_keeps_bundled_position_colors(tmp_path, topology_type):
    path = tagged_head(tmp_path, topology_type, invalid_metadata=True)
    importer, bm = import_surface(path)
    try:
        assert not importer._custom_topology
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


def tagged_head(tmp_path, topology_type, *, invalid_metadata=False):
    reader = get_dna_reader(TEST_DNA_FOLDER / "ada" / "head.dna")
    path = tmp_path / "head.dna"
    writer = get_dna_writer(path)
    try:
        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        writer.setMetaData(mesh_data.type_key("head"), topology_type)
        if invalid_metadata:
            writer.setMetaData(mesh_data.PREFIX + ".head", "invalid JSON must never be read for MetaHuman")
        writer.write()
    finally:
        release_dna_handle(writer)
        release_dna_handle(reader)
    return path


def test_custom_without_metadata_never_falls_back_even_with_default_vertex_count(tmp_path):
    path = tagged_head(tmp_path, "Custom")
    (tmp_path / "head_vertex_colors.json").write_text("invalid JSON must never be read")
    importer, bm = import_surface(path)
    try:
        assert importer._custom_topology
        assert not bm.loops.layers.color.keys()
    finally:
        bm.free()
        release_dna_handle(importer._dna_reader)
