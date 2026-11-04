"""Custom topology metadata survives native DNA and Blender merge/split paths."""

import json

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from test_vertex_color_roundtrip import write_surface

from character_dna.dna_io import (
    DNAExporter,
    DNAImporter,
    document,
    get_dna_reader,
    get_dna_writer,
    merged_graph,
    mesh_data,
    metadata,
    partitions,
    release_dna_handle,
)
from character_dna.editors.converter import scene_colors, scene_merge
from fixtures.rigs import RigProfile


@pytest.fixture
def surface():
    mesh = bpy.data.meshes.new("metadata_surface")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2), (0, 2, 3)])
    mesh.uv_layers.new()
    obj = bpy.data.objects.new("Custom_head_lod0_mesh", mesh)
    group = obj.vertex_groups.new(name="SEAM_neckline")
    group.add([0, 1, 2, 3], 1, "REPLACE")
    yield obj
    if obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.name in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)


def add_colors(mesh, domain, data_type, name="Color"):
    attr = mesh.color_attributes.new(name=name, domain=domain, type=data_type)
    colors = np.random.default_rng(99).random((len(attr.data), 4)).astype(np.float32)
    attr.data.foreach_set("color", colors.reshape(-1))
    mesh.color_attributes.active_color = attr
    mesh.color_attributes.render_color_index = mesh.color_attributes.find(name)
    return scene_colors.snapshot(mesh)


def export_surface(surface, source, target, kind="head"):
    instance = SimpleNamespace(
        name="Custom",
        editor_component=kind,
        head_mesh=surface,
        body_mesh=surface,
        head_rig=None,
        body_rig=None,
        head_dna_file_path=str(source),
        body_dna_file_path=str(source),
        get=lambda *_: None,
        output=SimpleNamespace(
            folder_path=str(target.parent),
            format="binary",
            head_topology_type="3D Scan Store",
            body_topology_type="3D Scan Store",
        ),
    )
    exporter = DNAExporter(instance, 100, component_type=kind, file_name=target.name)
    try:
        surface.name = f"Custom_{kind}_lod0_mesh"
        exporter._export_mesh(surface, 0)
        exporter._export_component_extensions()
        exporter._dna_writer.write()
    finally:
        release_dna_handle(exporter._dna_writer)
        release_dna_handle(exporter._dna_reader)


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("domain", ["POINT", "CORNER"])
@pytest.mark.parametrize("data_type", ["BYTE_COLOR", "FLOAT_COLOR"])
def test_embedded_export_import_without_sidecars(tmp_path, surface, kind, domain, data_type):
    expected = add_colors(surface.data, domain, data_type)
    source = write_surface(tmp_path / "source.dna", component=kind, lod_count=4 if kind == "body" else 8)
    target = tmp_path / f"{kind}.dna"
    export_surface(surface, source, target, kind)
    assert not list(tmp_path.glob("*_vertex_colors.json"))
    # A conflicting sidecar must not win over embedded data.
    (tmp_path / f"{kind}_vertex_colors.json").write_text("invalid JSON ignored")
    instance = SimpleNamespace(
        name="Reimported", output=SimpleNamespace(), head_dna_file_path=str(target), body_dna_file_path=str(target)
    )
    options = SimpleNamespace(
        import_vertex_colors=True,
        import_vertex_groups=False,
        import_normals=False,
        **{f"import_lod{i}": True for i in range(8)},
    )
    importer = DNAImporter(instance, options, 0.01, component_type=kind)
    imported = None
    try:
        assert mesh_data.type_name(importer._dna_reader, kind) == "3D Scan Store"
        importer.initialize_dna_data()
        assert getattr(instance.output, f"{kind}_topology_type") == "3D Scan Store"
        imported = importer.create_mesh_object(0, f"{kind}_lod0_mesh")
        assert scene_colors.snapshot(imported.data) == expected
        assert set(mesh_data.object_boundary(imported)) == {0, 1, 2, 3}
        assert importer._get_lod_settings() == [(0, True)]
    finally:
        if imported is not None:
            mesh = imported.data
            bpy.data.objects.remove(imported, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        release_dna_handle(importer._dna_reader)


def test_metadata_namespaces_preserve_foreign_and_extensions(tmp_path):
    path = write_surface(tmp_path / "head.dna")
    reader = get_dna_reader(path)
    writer = get_dna_writer(tmp_path / "updated.dna")
    try:
        from character_dna.bindings import dna

        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        payloads = {
            "character_dna.shape_extensions": {"version": 1, "scene_bridges": []},
            "character_dna.mesh_data.head": {"schema": 1, "component": "head", "meshes": []},
        }
        metadata.write_many(
            writer, reader, payloads, {"foreign": "unchanged", mesh_data.type_key("head"): "3D Scan Store"}
        )
        writer.write()
    finally:
        release_dna_handle(writer)
        release_dna_handle(reader)
    reader = get_dna_reader(tmp_path / "updated.dna", data_layer="Descriptor")
    try:
        assert reader.getMetaDataValue("foreign") == "unchanged"
        for prefix, expected in payloads.items():
            assert metadata.decode(reader, prefix=prefix) == expected
    finally:
        release_dna_handle(reader)


def test_json_dna_preserves_structured_metadata(tmp_path):
    source = write_surface(tmp_path / "head.dna")
    reader = get_dna_reader(source)
    writer = get_dna_writer(tmp_path / "head.json", "json")
    payload = {"schema": 1, "component": "head", "meshes": [], "note": 'quoted "name"'}
    try:
        from character_dna.bindings import dna

        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        metadata.write_many(
            writer,
            reader,
            {"character_dna.mesh_data.head": payload},
            {mesh_data.type_key("head"): '3D Scan Store "custom"'},
        )
        writer.write()
    finally:
        release_dna_handle(writer)
        release_dna_handle(reader)
    json.loads((tmp_path / "head.json").read_text())
    reader = get_dna_reader(tmp_path / "head.json", "json")
    try:
        assert mesh_data.read(reader, "head") == payload
        assert mesh_data.type_name(reader, "head") == '3D Scan Store "custom"'
    finally:
        release_dna_handle(reader)


@pytest.mark.parametrize("lods", [(0,), (0, 2)])
def test_native_merge_split_metadata_and_sparse_lods(tmp_path, surface, compact_dna_asset, lods):
    add_colors(surface.data, "CORNER", "BYTE_COLOR")
    graphs = {}
    for kind in ("head", "body"):
        graph = merged_graph.from_bytes((compact_dna_asset / f"{kind}.dna").read_bytes())
        definition = graph["defn1.1"]
        selected = []
        rows = []
        records = []
        for lod in range(graph["desc1.1"]["lodCount"]):
            if lod not in lods:
                rows.append([])
                continue
            index = definition["meshNames"].index(f"{kind}_lod{lod}_mesh")
            selected.append(index)
            mesh = graph["geom1.1"]["meshes"][index]
            # Keep native behavior/channels and choose the real triangle's full
            # boundary as a tiny custom neck seam.
            record = {
                "mesh_index": index,
                "name": definition["meshNames"][index],
                "lods": [lod],
                "topology": mesh_data.topology(3, [[0, 1, 2]]),
                "neckline": [0, 1, 2],
                "colors": [],
            }
            mesh["positions"] = {"xs": [0, 2, 0], "ys": [150, 150, 151], "zs": [0, 0, 0]}
            color = scene_colors.snapshot(surface.data)[0]
            color["indices"] = mesh_data.pack([0, 1, 2], "<u4")
            color["values"] = mesh_data.pack(mesh_data.unpack(color["values"])[:3], "|u1")
            record["colors"] = [color]
            records.append(record)
            rows.append([index])
        definition["lodMeshMapping"] = merged_graph._lod_map(rows)
        if lods == (0,):
            # Real custom overwrite DNA can retain eight behavior LODs but
            # contain only one geometry mapping row.
            definition["lodMeshMapping"] = merged_graph._lod_map(rows[:1])
        mesh_data.set_graph_payload(
            graph, kind, {"schema": 1, "component": kind, "space": "component", "meshes": records}, "3D Scan Store"
        )
        graphs[kind] = graph
    native, indices = merged_graph.compose(graphs, "Custom")
    assert [s["lod"] for s in indices["surfaces"]] == list(lods)
    for lod in range(8):
        assert bool(merged_graph._lod_rows(native["defn1.1"]["lodMeshMapping"], [lod])[0]) == (lod in lods)
    ownership = partitions.describe(graphs, native, indices)
    # Modify the authoritative merged color metadata after composition.
    payload = mesh_data.read(mesh_data.graph_reader(native), "head")
    color = payload["meshes"][0]["colors"][0]
    values = mesh_data.unpack(color["values"]).copy()
    values[0] = [12, 34, 56, 78]
    color["values"] = mesh_data.pack(values, "|u1")
    mesh_data.set_graph_payload(native, "head", payload, "Changed Head Type")
    projected = partitions.project(native, indices, ownership)
    record = mesh_data.read(mesh_data.graph_reader(projected["head"]), "head")["meshes"][0]
    np.testing.assert_array_equal(mesh_data.unpack(record["colors"][0]["values"])[0], [12, 34, 56, 78])
    assert mesh_data.type_name(mesh_data.graph_reader(projected["head"]), "head") == "Changed Head Type"
    assert mesh_data.read(mesh_data.graph_reader(projected["body"]), "body") == mesh_data.read(
        mesh_data.graph_reader(graphs["body"]), "body"
    )
    paths = {}
    for kind, graph in graphs.items():
        paths[kind] = tmp_path / f"{kind}.dna"
        paths[kind].write_bytes(merged_graph.to_bytes(graph))
    destination = tmp_path / "combined.dna"
    document.merge_files(paths, destination, "Custom")
    split = document.split_files(destination, tmp_path / "split")
    for kind in ("head", "body"):
        reader = get_dna_reader(split[kind])
        try:
            assert mesh_data.read(reader, kind) == mesh_data.read(mesh_data.graph_reader(graphs[kind]), kind)
        finally:
            release_dna_handle(reader)


def test_seam_validation_and_topology_fingerprint(surface):
    boundary = mesh_data.object_boundary(surface)
    assert len(boundary) == 4
    points = np.asarray([v.co[:] for v in surface.data.vertices])
    match = mesh_data.match_boundaries(boundary, list(reversed(boundary)), points, points, tolerance=0.001)
    assert all(a == b for a, b in match.items())
    with pytest.raises(ValueError, match="same vertex count"):
        mesh_data.match_boundaries(boundary, boundary[:3], points, points, tolerance=0.001)
    with pytest.raises(ValueError, match="separation"):
        mesh_data.match_boundaries(boundary, boundary, points, points + 1, tolerance=0.001)
    group = surface.vertex_groups["SEAM_neckline"]
    group.remove([1])
    with pytest.raises(ValueError, match="closed"):
        mesh_data.object_boundary(surface)
    record = {"name": "test", "topology": mesh_data.topology(4, [[0, 1, 2], [0, 2, 3]]), "colors": []}
    with pytest.raises(ValueError, match="topology mismatch"):
        mesh_data.validate_record(record, mesh_data.topology(4, [[0, 2, 1], [0, 2, 3]]))


def test_scene_topology_allows_only_unused_trailing_positions(surface):
    reader = SimpleNamespace(
        getVertexPositionCount=lambda _: 6,
        getFaceCount=lambda _: 2,
        getFaceVertexLayoutIndices=lambda _, face: [[0, 1, 2], [0, 2, 3]][face],
        getVertexLayoutPositionIndices=lambda _: [0, 1, 2, 3],
    )
    target = SimpleNamespace(dna_reader=reader)
    scene_merge._validate_topology(target, surface, 0)
    reader.getVertexLayoutPositionIndices = lambda _: [0, 1, 2, 5]
    with pytest.raises(ValueError, match="canonical DNA"):
        scene_merge._validate_topology(target, surface, 0)
    reader.getVertexLayoutPositionIndices = lambda _: [0, 2, 1, 3]
    with pytest.raises(ValueError, match="reordered DNA"):
        scene_merge._validate_topology(target, surface, 0)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_custom_calibration_resolves_embedded_neckline(tmp_path, surface, kind):
    from character_dna.dna_io import DNACalibrator

    readers = {}
    try:
        for component in ("head", "body"):
            source = write_surface(tmp_path / (component + "_source.dna"), component=component)
            path = tmp_path / (component + ".dna")
            export_surface(surface, source, path, component)
            readers[component] = get_dna_reader(path)
        other = "body" if kind == "head" else "head"
        calibrator = object.__new__(DNACalibrator)
        calibrator._dna_reader = readers[kind]
        calibrator._component_type = kind
        calibrator._seam_follower = kind
        calibrator._seam_reference_dna_path = None
        calibrator._seam_reference_reader = lambda: readers[other]
        name, pairs = calibrator._resolve_seam_pairs(f"{kind}_lod0_mesh", {})
        assert name == f"{other}_lod0_mesh"
        assert dict(pairs) == {i: i for i in range(4)}
        assert calibrator._resolve_seam_pairs("teeth_lod0_mesh", {}) is None
    finally:
        for reader in readers.values():
            release_dna_handle(reader)


def test_scene_merge_colors_and_edited_split(surface):
    add_colors(surface.data, "CORNER", "BYTE_COLOR")
    other_mesh = surface.data.copy()
    other = bpy.data.objects.new("other", other_mesh)
    merged_mesh = bpy.data.meshes.new("merged")
    merged_mesh.from_pydata(
        [v.co[:] for v in surface.data.vertices], [], [list(p.vertices) for p in surface.data.polygons] * 2
    )
    proxy = bpy.data.objects.new("proxy", merged_mesh)
    sources = {"head": surface, "body": other}
    baseline = None
    try:
        scene_colors.merge(merged_mesh, sources)
        baseline = merged_mesh.copy()
        proxy["dna_merge_baseline"] = baseline
        assert not scene_colors.prepare(proxy, sources)
        merged_mesh.color_attributes["Color"].data[0].color_srgb = (1, 0, 0, 0.5)
        changes = scene_colors.prepare(proxy, sources)
        assert set(changes) == {surface}
        scene_colors.restore(surface.data, changes[surface])
        assert surface.data.color_attributes["Color"].data[0].color_srgb[0] == 1
        assert surface.data.color_attributes["Color"].data[0].color_srgb[1] == 0
    finally:
        bpy.data.objects.remove(proxy)
        bpy.data.objects.remove(other)
        bpy.data.meshes.remove(merged_mesh)
        bpy.data.meshes.remove(other_mesh)
        if baseline:
            bpy.data.meshes.remove(baseline)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_full_export_writes_colors_only_inside_dna(tmp_path, rig_factory, compact_dna, kind):
    with rig_factory.lease(RigProfile(asset=str(compact_dna)), tmp_path) as instance:
        instance.output.method = "overwrite"
        instance.output.run_validations = False
        setattr(instance.output, f"{kind}_topology_type", "Custom")
        expected = add_colors(getattr(instance, f"{kind}_mesh").data, "CORNER", "BYTE_COLOR")
        target = tmp_path / "metadata-export"
        target.mkdir()
        instance.output.folder_path = str(target)
        exporter = DNAExporter(instance, 100, component_type=kind, file_name=f"{kind}.dna", textures=False)
        try:
            assert exporter.run()[0]
        finally:
            release_dna_handle(exporter._dna_writer)
            release_dna_handle(exporter._dna_reader)
        assert [p.name for p in target.iterdir()] == [f"{kind}.dna"]
        reader = get_dna_reader(target / f"{kind}.dna")
        try:
            assert mesh_data.read(reader, kind)["meshes"][0]["colors"] == expected
        finally:
            release_dna_handle(reader)


@pytest.mark.parametrize(
    "dtype,values", [("<u4", [-1]), ("<u4", [2**32]), ("<u4", [1.5]), ("|u1", [256]), ("<f4", [1e40])]
)
def test_metadata_array_rejects_overflow(dtype, values):
    with pytest.raises(ValueError, match="range"):
        mesh_data.pack(values, dtype)


def test_large_color_array_uses_bounded_fields_and_native_roundtrip(tmp_path):
    from character_dna.bindings import dna

    data = np.random.default_rng(7).random((100000, 4)).astype(np.float32)
    payload = {"colors": mesh_data.pack(data, "<f4")}
    encoded = metadata.encode(payload, prefix="test.colors")
    header = json.loads(encoded["test.colors"])
    assert header["chunks"] > 1
    assert all(len(value) <= metadata.CHUNK_BYTES for key, value in encoded.items() if key != "test.colors")
    source = get_dna_reader(write_surface(tmp_path / "source.dna"))
    writer = get_dna_writer(tmp_path / "colors.dna")
    try:
        writer.setFrom(source, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        metadata.write_many(writer, source, {"test.colors": payload})
        writer.write()
    finally:
        release_dna_handle(writer)
        release_dna_handle(source)
    reader = get_dna_reader(tmp_path / "colors.dna", data_layer="Descriptor")
    try:
        result = metadata.decode(reader, prefix="test.colors")
        np.testing.assert_array_equal(mesh_data.unpack(result["colors"]), data)
    finally:
        release_dna_handle(reader)
