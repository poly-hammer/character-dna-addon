"""Wrap materials follow corresponding faces and preserve input datablocks."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.editors.converter import correspondence, materials
from character_dna.editors.converter.pipeline import ConversionPipeline


@pytest.fixture
def scene_meshes():
    objects, meshes, created_materials = [], [], []

    def create(name, points, faces):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(points, [], faces)
        obj = bpy.data.objects.new(name, mesh)
        objects.append(obj)
        meshes.append(mesh)
        return obj

    def material(name):
        result = bpy.data.materials.new(name)
        created_materials.append(result)
        return result

    yield create, material
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        bpy.data.meshes.remove(mesh)
    for material in created_materials:
        bpy.data.materials.remove(material)


def test_materials_follow_reordered_faces_and_resolve_object_overrides(scene_meshes):
    create, material = scene_meshes
    points = [(x, y, 0) for y in range(2) for x in range(3)]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    permutation = np.array([5, 2, 1, 0, 3, 4])
    target = create("Converted", points, faces)
    source_points = np.empty((6, 3))
    source_points[permutation] = points
    source = create("Wrap", source_points, [permutation[f].tolist() for f in faces[::-1]])
    for obj in (source, target):
        uv = [tuple(obj.data.vertices[v].co[:2]) for face in obj.data.polygons for v in face.vertices]
        obj.data.uv_layers.new().data.foreach_set("uv", np.asarray(uv).ravel())
    first, second, override, previous = [material(name) for name in ("First", "Second", "Override", "Previous")]
    source.data.materials.append(first)
    source.data.materials.append(second)
    source.data.materials.append(None)
    source.material_slots[1].link = "OBJECT"
    source.material_slots[1].material = override
    source.data.polygons[0].material_index = 1
    source.data.polygons[1].material_index = 0
    target.data.materials.append(previous)
    target.material_slots[0].link = "OBJECT"
    target.material_slots[0].material = previous
    before_uv = [tuple(loop.uv) for loop in target.data.uv_layers.active.data]

    mapping, _ = correspondence.match(source.data, target.data)
    materials.transfer(source, target, mapping)

    assert list(target.data.materials) == [first, override, None]
    assert all(slot.link == "DATA" for slot in target.material_slots)
    assert [face.material_index for face in target.data.polygons] == [0, 1]
    assert [tuple(loop.uv) for loop in target.data.uv_layers.active.data] == before_uv
    assert list(source.data.materials) == [first, second, None]
    assert source.material_slots[1].link == "OBJECT"
    assert source.material_slots[1].material == override
    assert [face.material_index for face in source.data.polygons] == [1, 0]


def test_component_subset_uses_full_body_face_assignments(scene_meshes):
    create, material = scene_meshes
    points = [(x, y, 0) for y in range(2) for x in range(3)]
    source = create("Full Body", points, [[1, 2, 5, 4], [0, 1, 4, 3]])
    first, second = material("Head"), material("Body")
    source.data.materials.append(first)
    source.data.materials.append(second)
    source.data.polygons[0].material_index = 1
    source.data.polygons[1].material_index = 0
    # The component vertex domain is smaller than the merged input domain.
    ids = np.array([0, 1, 3, 4])
    component = create("Component", np.array(points)[ids], [[0, 1, 3, 2]])
    materials.transfer(source, component, ids)
    assert list(component.data.materials) == [first, second]
    assert component.data.polygons[0].material_index == 0


@pytest.mark.parametrize("method,enabled", [("WRAP", True), ("WRAP", False), ("RELATIVE", True)])
def test_transfer_is_opt_in_for_wrap_relationships(scene_meshes, method, enabled):
    create, material = scene_meshes
    points, faces = [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]]
    source, target = create("Input", points, faces), create("Output", points, faces)
    original, replacement = material("Original"), material("Replacement")
    target.data.materials.append(original)
    source.data.materials.append(replacement)
    pipeline = SimpleNamespace(
        rows={"head": {"method": method, "transfer_material": enabled}}, objects={"head": target}
    )
    ConversionPipeline._transfer_materials(pipeline, "head", source, np.arange(3))
    assert target.data.materials[0] == (replacement if method == "WRAP" and enabled else original)


def test_target_without_materials_clears_converted_slots(scene_meshes):
    create, material = scene_meshes
    points, faces = [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]]
    source, target = create("Empty", points, faces), create("Converted", points, faces)
    target.data.materials.append(material("Template"))
    materials.transfer(source, target, np.arange(3))
    assert len(target.material_slots) == 0
    assert target.data.polygons[0].material_index == 0
