"""Side inputs preserve topology, UVs, material regions and original datablocks."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.editors.converter import mesh_parts
from character_dna.editors.converter.pipeline import ConversionPipeline, positions


@pytest.fixture
def meshes():
    original_meshes, original_objects, original_materials = map(
        set, (bpy.data.meshes, bpy.data.objects, bpy.data.materials)
    )

    def create(name, points, faces):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(points, [], faces)
        return bpy.data.objects.new(name, mesh)

    yield create
    for obj in set(bpy.data.objects) - original_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in set(bpy.data.meshes) - original_meshes:
        bpy.data.meshes.remove(mesh)
    for material in set(bpy.data.materials) - original_materials:
        bpy.data.materials.remove(material)


def test_shared_sides_copy_positions_and_material_regions_from_original_vertex_ids(meshes):
    # Interleaved vertex IDs, reordered faces, and a center offset from X=0.
    left = np.array([(11, 0, 0), (12, 0, 0), (12, 1, 0), (11, 1, 0)], dtype=np.float32)
    right = left - (3, 0, 0)
    points = np.array([p for pair in zip(left, right, strict=True) for p in pair])
    source = meshes("Paired inputs", points, [[1, 5, 7], [0, 2, 4], [1, 3, 5], [0, 4, 6]])
    vertex_uvs = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
    uv = source.data.uv_layers.new()
    uv.data.foreach_set("uv", np.array([vertex_uvs[loop.vertex_index // 2] for loop in source.data.loops]).ravel())
    originals = []
    for index, face in enumerate(source.data.polygons):
        material = bpy.data.materials.new(f"Part material {index}")
        material.diffuse_color = (index / 4, 0, 0, 1)
        source.data.materials.append(material)
        face.material_index = index
        originals.append(material)
    rows, inputs, outputs = [], [], {}
    for role, part, dna_name in (("left", "LEFT", "eyeLeft_lod0_mesh"), ("right", "RIGHT", "eyeRight_lod0_mesh")):
        rows.append({"id": role, "method": "COPY", "mesh_part": part, "mesh": dna_name})
        inputs.append(SimpleNamespace(role=role, fit_method="COPY", scene_object=source, material_source=source))
        output = meshes(role, left * 0.1, [[0, 1, 2], [0, 2, 3]])
        output.data.uv_layers.new().data.foreach_set("uv", vertex_uvs[[0, 1, 2, 0, 2, 3]].ravel())
        outputs[role] = output
    operator = SimpleNamespace(
        _properties=SimpleNamespace(extra_meshes=inputs, new_name="Parts", validate_uvs=True, uv_tolerance=1e-5),
        _owned={"materials": set()},
    )
    pipeline = ConversionPipeline(operator, bpy.context, {"meshes": rows})
    pipeline.objects = outputs
    before = set(bpy.data.meshes)
    pipeline._supplied().apply(1)
    for role, expected, face_indices in (("left", left, [1, 3]), ("right", right, [2, 0])):
        output = outputs[role]
        np.testing.assert_array_equal(positions(output), expected)
        for face, source_index in zip(output.data.polygons, face_indices, strict=True):
            copied = output.data.materials[face.material_index]
            assert copied != originals[source_index]
            assert tuple(copied.diffuse_color) == tuple(originals[source_index].diffuse_color)
            assert copied.name.startswith(f"Parts_eye{role.title()}_shader")
    np.testing.assert_array_equal(positions(source), points)
    assert list(source.data.materials) == originals
    assert [face.material_index for face in source.data.polygons] == [0, 1, 2, 3]
    assert set(bpy.data.meshes) == before


@pytest.mark.parametrize("part,expected", [("LEFT", [1, 2, 4, 5]), ("RIGHT", [0, 1, 3, 4])])
def test_center_vertices_are_shared_and_crossing_faces_are_removed(meshes, part, expected):
    obj = meshes(
        "Centered", [(x, y, 0) for y in range(2) for x in (9, 10, 11)], [[0, 1, 4, 3], [1, 2, 5, 4], [0, 2, 5, 3]]
    )
    before = set(bpy.data.meshes)
    with mesh_parts.extract(obj.data, part) as (subset, ids):
        assert ids.tolist() == expected
        assert len(subset.vertices) == 4 and len(subset.polygons) == 1
    assert len(obj.data.vertices) == 6 and len(obj.data.polygons) == 3
    assert set(bpy.data.meshes) == before


def test_full_passes_through_and_failed_match_cleans_temporary_mesh(meshes):
    source = meshes("Input", [(x, y, 0) for y in range(2) for x in (9, 10, 11)], [[0, 1, 4, 3], [1, 2, 5, 4]])
    target = meshes("Wrong topology", [(0, 0, 0)] * 3, [[0, 1, 2]])
    before = set(bpy.data.meshes)
    with mesh_parts.extract(source.data) as (mesh, ids):
        assert mesh == source.data and ids.tolist() == list(range(6))
    props = SimpleNamespace(
        extra_meshes=[SimpleNamespace(role="eye", fit_method="COPY", scene_object=source, material_source=None)],
        validate_uvs=False,
    )
    pipeline = ConversionPipeline(
        SimpleNamespace(_properties=props),
        bpy.context,
        {
            "meshes": [
                {"id": "eye", "method": "COPY", "mesh": "eyeLeft_lod0_mesh", "mesh_part": "LEFT"},
            ]
        },
    )
    pipeline.objects = {"eye": target}
    with pytest.raises(ValueError, match=r"eyeLeft_lod0_mesh.*LEFT.*counts"):
        pipeline._supplied()
    assert set(bpy.data.meshes) == before


def test_empty_side_input_reports_error_without_leaking_mesh(meshes):
    source = meshes("Empty", [], [])
    before = set(bpy.data.meshes)
    with pytest.raises(ValueError, match="empty mesh"), mesh_parts.extract(source.data, "RIGHT"):
        pytest.fail("Empty side input should fail")
    assert set(bpy.data.meshes) == before
