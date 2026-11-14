"""Arbitrary accessory topology survives conversion, DNA export, and scene creation."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.dna_io import mesh_data, release_dna_handle
from character_dna.dna_io.importer import DNAImporter
from character_dna.dna_io.misc import get_dna_reader
from character_dna.editors.converter import templates
from character_dna.editors.converter.native.template import TemplateReader
from character_dna.editors.converter.properties import get_converter_properties


def make_object(name, points, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def assert_standard_import(path, expected):
    instance = SimpleNamespace(name="Reopened", head_dna_file_path=str(path), output=SimpleNamespace())
    options = SimpleNamespace(
        import_vertex_groups=True,
        import_vertex_colors=False,
        import_normals=True,
        **{f"import_lod{i}": True for i in range(8)},
    )
    importer = DNAImporter(instance, options, 0.01)
    try:
        importer.initialize_dna_data()
        obj = importer.create_mesh_object(0, "eyelashes_lod0_mesh")
        np.testing.assert_allclose([v.co[:] for v in obj.data.vertices], expected, atol=2e-6, rtol=0)
        assert [layer.name for layer in obj.data.uv_layers] == ["DiffuseUV", "Detail"]
        assert all(sum(g.weight for g in v.groups) == pytest.approx(1, abs=1e-6) for v in obj.data.vertices)
    finally:
        release_dna_handle(importer._dna_reader)


def assert_replacement_dna(path, names):
    reader = get_dna_reader(path)
    try:
        records = {r["name"]: r for r in mesh_data.read(reader, "head")["meshes"]}
        for name in names:
            index = next(i for i in range(reader.getMeshCount()) if reader.getMeshName(i) == name)
            assert reader.getVertexPositionCount(index) == 8
            assert reader.getSkinWeightsCount(index) == 8
            mesh_data.validate_record(records[name], mesh_data.topology(8, mesh_data.reader_faces(reader, index)))
            lods = [lod for lod in range(reader.getLODCount()) if index in reader.getMeshIndicesForLOD(lod)]
            allowed = set.intersection(*(set(reader.getJointIndicesForLOD(lod)) for lod in lods))
            allowed |= set(range(reader.getJointCount())) - set(reader.getJointIndicesForLOD(0))
            for vertex in range(8):
                assert sum(reader.getSkinWeightsValues(index, vertex)) == pytest.approx(1, abs=1e-6)
                assert set(reader.getSkinWeightsJointIndices(index, vertex)) <= allowed
            for target in range(reader.getBlendShapeTargetCount(index)):
                assert len(reader.getBlendShapeTargetVertexIndices(index, target)) == 0
    finally:
        release_dna_handle(reader)


@pytest.mark.parametrize("filename", ["3d_scan_store_head_basic.json", "metahuman_head_basic.json"])
def test_replacement_round_trip_keeps_source_data_and_valid_lod_weights(empty_character_scene, tmp_path, filename):
    props = get_converter_properties()
    templates.load_template(props, templates.TEMPLATE_FOLDER / filename)
    template = TemplateReader(str(templates.current_dna_files(props)["head"]), "head")
    head = template.meshes["head_lod0_mesh"]
    faces = [face.tolist() for face in np.split(head.corners, np.cumsum(head.sizes)[:-1])]
    source_head = make_object("Input Head", head.positions, faces)
    source_head.data.uv_layers.new().data.foreach_set("uv", head.uv.ravel())
    rows = {item.role: item for item in props.extra_meshes}
    rows["head"].scene_object = source_head
    row = rows["eyelashes"]
    row.fit_method = "REPLACE"
    row.mesh_part = "FULL"
    for reference in row.references:
        reference.enabled = reference.key == "head"
    anchor = template.meshes[row.dna_mesh_name].positions[0]
    points = anchor + np.array(
        [
            [0, 0, 0],
            [0.001, 0, 0],
            [0.001, -0.003, 0.001],
            [0, -0.003, 0.001],
            [0.003, 0, 0],
            [0.004, 0, 0],
            [0.004, -0.003, 0.001],
            [0.003, -0.003, 0.001],
        ]
    )
    source = make_object("Supplied Cards", points, [[0, 1, 2, 3], [4, 5, 6, 7]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [3, 0], [3, 1], [2, 1]], dtype=np.float32)
    source.data.uv_layers.new(name="DiffuseUV").data.foreach_set("uv", uv.ravel())
    source.data.uv_layers.new(name="Detail").data.foreach_set("uv", (uv * 0.4).ravel())
    source.data.uv_layers[0].active_render = True
    root_group = source.vertex_groups.new(name="Roots")
    root_group.add([0, 1, 4, 5], 1, "REPLACE")
    for color in ((0.1, 0.3, 0.5, 1), (0.8, 0.2, 0.1, 1)):
        material = bpy.data.materials.new("Card Material")
        material.diffuse_color = color
        source.data.materials.append(material)
    source.data.polygons[1].material_index = 1
    row.scene_object, row.root_group = source, root_group.name
    props.new_name, props.new_folder, props.zero_shape_deltas = "Replaced", str(tmp_path), True
    family = next(r["lods"] for r in templates.from_properties(props)["meshes"] if r["id"] == row.role)
    family = [name for name in family if name in template.info["mesh_names"]]
    assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
    result = bpy.data.objects["Replaced_eyelashes_lod0_mesh"]
    positions = np.empty((8, 3), dtype=np.float32)
    result.data.vertices.foreach_get("co", positions.ravel())
    np.testing.assert_allclose(positions, points, atol=2e-6, rtol=0)
    assert [list(p.vertices) for p in result.data.polygons] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [p.material_index for p in result.data.polygons] == [0, 1]
    for original, copied in zip(source.data.materials, result.data.materials, strict=True):
        np.testing.assert_allclose(original.diffuse_color, copied.diffuse_color)
    assert [layer.name for layer in result.data.uv_layers] == ["DiffuseUV", "Detail"]
    for layer, expected in zip(result.data.uv_layers, [uv, uv * 0.4], strict=True):
        actual = np.empty((8, 2), dtype=np.float32)
        layer.data.foreach_get("uv", actual.ravel())
        np.testing.assert_allclose(actual, expected)
    assert_replacement_dna(tmp_path / "head.dna", [row.dna_mesh_name, *family])
    assert len(result.vertex_groups) > 0
    assert all(any(g.weight > 0 for g in v.groups) for v in result.data.vertices)
    assert source.hide_get() and source.hide_render
    assert_standard_import(tmp_path / "head.dna", points)


def test_full_body_advanced_replacements_follow_pose_normalization(empty_character_scene, tmp_path):
    props = get_converter_properties()
    templates.load_template(props, templates.TEMPLATE_FOLDER / "3d_scan_store_full_body.json")
    readers = {kind: TemplateReader(str(path), kind) for kind, path in templates.current_dna_files(props).items()}
    props.input_layout, props.scene_layout = "SPLIT", "SPLIT"
    replacements = {}
    for item in templates.input_rows(props):
        shape = readers[item.component].meshes[item.dna_mesh_name]
        item.mesh_part = "FULL"
        points = shape.positions
        faces = [face.tolist() for face in np.split(shape.corners, np.cumsum(shape.sizes)[:-1])]
        if item.fit_method == "REPLACE":
            points = points[0] + np.array([[0, 0, 0], [0.001, 0, 0], [0.001, -0.003, 0.001], [0, -0.003, 0.001]])
            faces = [[0, 1, 2, 3]]
            replacements[item.dna_mesh_name] = points
        obj = make_object(item.role, points, faces)
        if item.fit_method == "COPY":
            obj.data.uv_layers.new().data.foreach_set("uv", shape.uv.ravel())
        item.scene_object = obj
    props.new_name, props.new_folder = "BodyReplacement", str(tmp_path)
    assert props.pose_normalization == "LIMBS"
    assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
    assert (tmp_path / "head.dna").is_file() and (tmp_path / "body.dna").is_file()
    for name, expected in replacements.items():
        obj = bpy.data.objects[f"BodyReplacement_{name}"]
        np.testing.assert_allclose([v.co[:] for v in obj.data.vertices], expected, atol=5e-5, rtol=0)
        assert all(sum(g.weight for g in v.groups) == pytest.approx(1, abs=1e-6) for v in obj.data.vertices)


def test_replacement_recipe_requires_one_supplied_reference_and_full_mesh(empty_character_scene):
    props = get_converter_properties()
    templates.load_template(props, templates.TEMPLATE_FOLDER / "3d_scan_store_head_advanced.json")
    rows = {item.role: item for item in props.extra_meshes}
    row = rows["eyelashes"]
    row.root_group, row.attachment_group = "Roots", "PRESERVE_LidRegion"
    recipe = templates.from_properties(props)
    templates.apply_template(props, recipe)
    row = next(item for item in props.extra_meshes if item.role == "eyelashes")
    assert (row.root_group, row.attachment_group) == ("Roots", "PRESERVE_LidRegion")
    row.mesh_part = "LEFT"
    with pytest.raises(ValueError, match="Full mesh part"):
        templates.validate_template(templates.from_properties(props))
    row.mesh_part = "FULL"
    for reference in row.references:
        reference.enabled = reference.key == "brows"
    with pytest.raises(ValueError, match="Copy to Target reference"):
        templates.validate_template(templates.from_properties(props))
