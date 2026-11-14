"""Named UV layers retain independent seams through native DNA round trips."""

import copy

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from test_vertex_color_roundtrip import write_surface

from character_dna.dna_io import DNACalibrator, DNAExporter, DNAImporter, get_dna_reader, mesh_data, release_dna_handle
from fixtures.rigs import RigProfile


def snapshot(mesh):
    result = []
    for layer in mesh.uv_layers:
        values = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        layer.data.foreach_get("uv", values)
        result.append((layer.name, layer == mesh.uv_layers.active, layer.active_render, values.tolist()))
    return result


@pytest.fixture
def surface():
    mesh = bpy.data.meshes.new("uv_roundtrip")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2), (0, 2, 3)])
    obj = bpy.data.objects.new("Original_head_lod0_mesh", mesh)
    # The native map is deliberately not first, and secondary seams differ from it.
    for name, values in (
        ("HeadOnly", [(-1, 0.1), (2.3, 0.2), (2.3, 1.1), (0.4, 0.3), (0.8, 1.6), (-0.1, 2.3)]),
        ("DiffuseUV", [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)]),
        ("Detail UV", [(3.1, 0), (4, 0), (4, 1), (5, 0), (6, 1), (5, 1)]),
    ):
        layer = mesh.uv_layers.new(name=name)
        layer.data.foreach_set("uv", np.array(values, dtype=np.float32).ravel())
    mesh.uv_layers.active = mesh.uv_layers["HeadOnly"]
    mesh.uv_layers["DiffuseUV"].active_render = True
    yield obj
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)


def write_mesh(obj, source, path, kind, topology_type):
    obj.name = f"Original_{kind}_lod0_mesh"
    instance = SimpleNamespace(
        name="Original",
        head_mesh=obj,
        body_mesh=obj,
        head_rig=None,
        body_rig=None,
        get=lambda *_: None,
        head_dna_file_path=str(source),
        body_dna_file_path=str(source),
        output=SimpleNamespace(
            folder_path=str(path.parent),
            format="json" if path.suffix == ".json" else "binary",
            head_topology_type=topology_type,
            body_topology_type=topology_type,
        ),
    )
    reader = get_dna_reader(source, "json" if source.suffix == ".json" else "binary")
    exporter = DNAExporter(
        instance, 100, component_type=kind, file_name=path.name, vertex_groups=False, vertex_colors=False, reader=reader
    )
    try:
        exporter._export_mesh(obj, 0)
        exporter._export_component_extensions()
        exporter._dna_writer.write()
    finally:
        release_dna_handle(exporter._dna_writer)
        release_dna_handle(exporter._dna_reader)


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("topology_type", ["MetaHuman", "3D Scan Store"])
@pytest.mark.parametrize("primary_name", ["DiffuseUV", "Base UV"])
@pytest.mark.parametrize("suffix", [".dna", ".json"])
def test_named_uv_roundtrip_and_native_primary(tmp_path, surface, kind, topology_type, primary_name, suffix):
    surface.data.uv_layers["DiffuseUV"].name = primary_name
    expected = snapshot(surface.data)
    source = write_surface(tmp_path / "source.dna", component=kind)
    target = tmp_path / (kind + suffix)
    write_mesh(surface, source, target, kind, topology_type)
    assert snapshot(surface.data) == expected  # Export must not change the selected layer.
    options = SimpleNamespace(
        import_vertex_groups=False,
        import_vertex_colors=False,
        import_normals=False,
        **{f"import_lod{i}": True for i in range(8)},
    )
    instance = SimpleNamespace(
        name="Imported", head_dna_file_path=str(target), body_dna_file_path=str(target), output=SimpleNamespace()
    )
    importer = DNAImporter(instance, options, 0.01, component_type=kind)
    obj = None
    try:
        importer.initialize_dna_data()
        reader = importer._dna_reader
        uv_ids = reader.getVertexLayoutTextureCoordinateIndices(0)
        uv = np.array(
            list(zip(reader.getVertexTextureCoordinateUs(0), reader.getVertexTextureCoordinateVs(0), strict=True))
        )
        actual = [
            uv[uv_ids[i]] for face in range(reader.getFaceCount(0)) for i in reader.getFaceVertexLayoutIndices(0, face)
        ]
        native_expected = np.array(expected[1][3]).reshape(-1, 2)
        corners = mesh_data.corner_order(surface.data, mesh_data.reader_faces(reader, 0), list(range(4)))
        np.testing.assert_array_equal(actual, native_expected[corners])
        obj = importer.create_mesh_object(0, f"{kind}_lod0_mesh")
        assert snapshot(obj.data) == expected
        # Deleting/renaming an extra map is authoritative on the next export.
        obj.data.uv_layers.remove(obj.data.uv_layers["Detail UV"])
        obj.data.uv_layers["HeadOnly"].name = "Head Detail"
        reexport = tmp_path / "reexport.dna"
        write_mesh(obj, target, reexport, kind, topology_type)
        check = get_dna_reader(reexport)
        try:
            layers = mesh_data.record_for(check, kind, 0)["uv_layers"]
            assert [layer["name"] for layer in layers] == ["Head Detail", primary_name]
        finally:
            release_dna_handle(check)
    finally:
        if obj is not None:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        release_dna_handle(importer._dna_reader)


@pytest.mark.parametrize("damage", ["name", "primary", "active", "indices", "shape", "dtype"])
def test_invalid_uv_metadata_is_rejected(surface, damage):
    record = mesh_data.capture(surface, 0, "head_lod0_mesh", 4, [[0, 1, 2], [0, 2, 3]], [0], colors=False)
    record = copy.deepcopy(record)
    layer = record["uv_layers"][0]
    if damage == "name":
        layer["name"] = record["uv_layers"][1]["name"]
    elif damage in {"primary", "active"}:
        layer[damage] = not layer[damage]
    elif damage == "indices":
        layer["indices"] = mesh_data.pack([0, 0, 2, 3, 4, 5], "<u4")
    elif damage == "shape":
        layer["values"] = mesh_data.pack(np.zeros((6, 3)), "<f4")
    else:
        layer["values"] = mesh_data.pack(np.zeros((6, 2)), "<u4")
    with pytest.raises(ValueError, match="UV"):
        mesh_data.validate_record(record, record["topology"])


def test_fitting_and_mesh_subsets_use_native_uvs_with_extra_layer_selected(surface):
    from character_dna.editors.converter import correspondence
    from character_dna.editors.converter.mesh import parts as mesh_parts
    from character_dna.editors.shared.utilities import _extract_blender_uv_layout

    original = snapshot(surface.data)
    expected = np.array(original[1][3]).reshape(-1, 2)
    with mesh_parts.subset(surface.data, np.arange(4)) as (subset, _):
        assert subset.uv_layers.active.name == "DiffuseUV"
        mapping, _ = correspondence.match(surface.data, subset, require_uvs=True)
        np.testing.assert_array_equal(mapping, np.arange(4))
    np.testing.assert_array_equal(_extract_blender_uv_layout(surface)[0], expected)
    assert snapshot(surface.data) == original


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("export_type", [DNAExporter, DNACalibrator])
def test_full_export_and_calibration_capture_extra_uvs(tmp_path, rig_factory, compact_dna, kind, export_type):
    with rig_factory.lease(RigProfile(asset=str(compact_dna)), tmp_path) as instance:
        instance.output.run_validations = False
        setattr(instance.output, f"{kind}_topology_type", "Custom")
        obj = getattr(instance, f"{kind}_mesh")
        layer = obj.data.uv_layers.new(name="HeadOnly")
        values = np.arange(len(obj.data.loops) * 2, dtype=np.float32).reshape(-1, 2) * 0.17 - 1
        layer.data.foreach_set("uv", values.ravel())
        obj.data.uv_layers.active = layer
        instance.output.folder_path = str(tmp_path)
        exporter = export_type(
            instance,
            100,
            component_type=kind,
            file_name="extra.dna",
            textures=False,
            vertex_groups=False,
            vertex_colors=False,
            shape_keys=False,
            seam_follower=None,
        )
        try:
            assert exporter.run()[0]
        finally:
            release_dna_handle(exporter._dna_writer)
            release_dna_handle(exporter._dna_reader)
        reader = get_dna_reader(tmp_path / "extra.dna")
        try:
            record = mesh_data.record_for(reader, kind, 0)
            extra = next(row for row in record["uv_layers"] if row["name"] == "HeadOnly")
            np.testing.assert_array_equal(mesh_data.unpack(extra["values"]), values)
        finally:
            release_dna_handle(reader)
