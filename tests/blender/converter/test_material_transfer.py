"""Copied materials follow corresponding faces and preserve input datablocks."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.components.base import CharacterComponentBase
from character_dna.editors.converter import correspondence, materials
from character_dna.editors.converter.pipeline import ConversionPipeline


@pytest.mark.parametrize("observed", [False, True])
def test_named_exclusion_uses_assigned_source_material_for_supplied_or_missing_region(scene_meshes, observed):
    from character_dna.editors.converter.native.materials import region_material_indices

    create, material = scene_meshes
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)]
    target = create("Joined DNA teeth", points, [[0, 1, 2], [3, 4, 5]])
    source = create("Teeth source", points, [[3, 4, 5], [0, 1, 2]])
    teeth, tongue = material("DentalSource"), material("Tongue")
    tongue.use_nodes = True
    source.data.materials.append(teeth)
    source.data.materials.append(tongue)
    source.data.polygons[0].material_index = 1
    mask = {f"EXCLUDE_{tongue.name}": [3, 4, 5]}
    before = set(bpy.data.materials)
    indices = region_material_indices(source, target, mask)
    materials.transfer(
        source, target, instance_name="Test", dna_mesh_name="teeth_lod0_mesh", exclusions=mask, face_indices=indices
    )
    materials.assign_exclusion_materials(target, mask, np.full(6, not observed), source)
    assert target.data.materials[target.data.polygons[1].material_index] == tongue
    primary = target.data.materials[target.data.polygons[0].material_index]
    assert primary != teeth and primary.name.startswith("Test_teeth_shader")
    assert set(bpy.data.materials) - before == {primary}
    assert list(source.data.materials) == [teeth, tongue]


def test_unmatched_exclusion_material_keeps_blank_fallback(scene_meshes):
    create, material = scene_meshes
    target = create("Excluded", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]])
    source = create("Source", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]])
    source.data.materials.append(material("Unrelated"))
    materials.assign_exclusion_materials(target, {"EXCLUDE_Optional": [0, 1, 2]}, np.ones(3, dtype=bool), source)
    result = target.data.materials[target.data.polygons[0].material_index]
    assert result.name.startswith("Optional") and result.use_nodes
    assert not any(node.type == "TEX_IMAGE" for node in result.node_tree.nodes)


@pytest.fixture
def scene_meshes():
    objects, meshes = [], []
    original_materials = set(bpy.data.materials)

    def create(name, points, faces):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(points, [], faces)
        obj = bpy.data.objects.new(name, mesh)
        objects.append(obj)
        meshes.append(mesh)
        return obj

    def material(name):
        result = bpy.data.materials.new(name)
        return result

    yield create, material
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        bpy.data.meshes.remove(mesh)
    for material in set(bpy.data.materials) - original_materials:
        bpy.data.materials.remove(material)


@pytest.mark.parametrize("dna_mesh_name", ["head_lod0_mesh", "unknown"])
def test_materials_follow_reordered_faces_and_resolve_object_overrides(scene_meshes, dna_mesh_name):
    create, material = scene_meshes
    points = [(x, y, 0) for y in range(2) for x in range(3)]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    permutation = np.array([5, 2, 1, 0, 3, 4])
    target = create("Converted", points, faces)
    source_points = np.empty((6, 3))
    source_points[permutation] = points
    source = create("Input", source_points, [permutation[f].tolist() for f in faces[::-1]])
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
    materials.transfer(source, target, mapping, instance_name="Test", dna_mesh_name=dna_mesh_name)

    if dna_mesh_name == "unknown":
        assert list(target.data.materials) == [first, override, None]
    else:
        assert [slot.material.name if slot.material else None for slot in target.material_slots] == [
            "Test_head_shader",
            "Test_head_shader.001",
        ]
        assert first not in list(target.data.materials) and override not in list(target.data.materials)
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


@pytest.mark.parametrize("method", ["COPY", "RELATIVE"])
@pytest.mark.parametrize("enabled", [True, False])
def test_transfer_is_opt_in_for_each_relationship(scene_meshes, method, enabled):
    create, material = scene_meshes
    points, faces = [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]]
    source, target = create("Input", points, faces), create("Output", points, faces)
    original, replacement = material("Original"), material("Replacement")
    target.data.materials.append(original)
    source.data.materials.append(replacement)
    pipeline = SimpleNamespace(
        rows={"head": {"method": method, "mesh": "unknown"}},
        objects={"head": target},
        material_sources={"head": source} if enabled else {},
        exclusions={},
        operator=SimpleNamespace(_properties=SimpleNamespace(new_name="Test"), _owned={"materials": set()}),
    )
    if method == "COPY":
        ConversionPipeline._transfer_materials(pipeline, "head", source, np.arange(3))
    else:
        ConversionPipeline._transfer_materials(pipeline, "head")
    assert target.data.materials[0] == (replacement if enabled else original)


def test_target_without_materials_clears_converted_slots(scene_meshes):
    create, material = scene_meshes
    points, faces = [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]]
    source, target = create("Empty", points, faces), create("Converted", points, faces)
    target.data.materials.append(material("Template"))
    materials.transfer(source, target, np.arange(3))
    assert len(target.material_slots) == 0
    assert target.data.polygons[0].material_index == 0


def test_standard_head_and_body_shaders_are_independent_copies(scene_meshes):
    create, material = scene_meshes
    points, faces = [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]]
    source, head, body = [create(name, points, faces) for name in ("Source", "Head", "Body")]
    original = material("Body")
    original.use_nodes = True
    original.node_tree.nodes.new("ShaderNodeTexImage").label = "Preserve source shader"
    source.data.materials.append(original)
    for target, dna_mesh_name in ((head, "head_lod0_mesh"), (body, "body_lod0_mesh")):
        materials.transfer(source, target, instance_name="Test", dna_mesh_name=dna_mesh_name)
    head_shader, body_shader = head.data.materials[0], body.data.materials[0]
    assert head_shader.name == "Test_head_shader"
    assert body_shader.name == "Test_body_shader"
    assert len({head_shader, body_shader, original}) == 3
    assert len({head_shader.node_tree, body_shader.node_tree, original.node_tree}) == 3
    assert head_shader.node_tree.nodes.get("Image Texture").label == "Preserve source shader"
    head_shader.node_tree.nodes.get("Image Texture").label = "Independent"
    assert body_shader.node_tree.nodes.get("Image Texture").label == "Preserve source shader"
    assert source.data.materials[0] == original
    assert original.name == "Body"


def test_merged_material_regions_get_only_their_own_named_shader(scene_meshes):
    create, material = scene_meshes
    points = [(x, y, 0) for y in range(2) for x in range(3)]
    source = create("Full Body", points, [[0, 1, 4, 3], [1, 2, 5, 4]])
    originals = [material("Skin head"), material("Skin body")]
    for original in originals:
        source.data.materials.append(original)
    source.data.polygons[1].material_index = 1
    for kind, ids in (("head", [0, 1, 3, 4]), ("body", [1, 2, 4, 5])):
        target = create(kind, np.array(points)[ids], [[0, 1, 3, 2]])
        materials.transfer(source, target, np.array(ids), instance_name="Test", dna_mesh_name=f"{kind}_lod0_mesh")
        assert [slot.material.name for slot in target.material_slots] == [f"Test_{kind}_shader"]
        assert target.data.polygons[0].material_index == 0
    assert list(source.data.materials) == originals
    assert [face.material_index for face in source.data.polygons] == [0, 1]


@pytest.mark.parametrize(
    "mesh_name,expected",
    [
        ("teeth_lod0_mesh", "teeth_shader"),
        ("eyeLeft_lod0_mesh", "eyeLeft_shader"),
        ("eyelashesShadow_lod0_mesh", "eyelashesShadow_shader"),
        ("brows_lod0_mesh", None),
    ],
)
def test_shader_names_follow_constants(mesh_name, expected):
    assert materials.shader_name(mesh_name) == expected


def test_independent_material_source_needs_no_topology_match_for_uniform_material(scene_meshes):
    create, material = scene_meshes
    points, faces = [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]]
    input_mesh, target = create("Input", points, faces), create("Target", points, faces)
    source = create("Material source", [*points, (1, 1, 0)], [[0, 1, 3, 2]])
    original = material("Source material")
    source.data.materials.append(original)
    instance = SimpleNamespace()
    owned = {"materials": set()}
    pipeline = SimpleNamespace(
        rows={"head": {"method": "COPY", "mesh": "head_lod0_mesh", "component": "head"}},
        objects={"head": target},
        material_sources={"head": source},
        exclusions={},
        operator=SimpleNamespace(_properties=SimpleNamespace(new_name="Test"), _rig_instance=instance, _owned=owned),
    )
    ConversionPipeline._transfer_materials(pipeline, "head", input_mesh, np.arange(3))
    assert target.data.materials[0].name == "Test_head_shader"
    assert instance.head_material == target.data.materials[0]
    assert owned["materials"] == {target.data.materials[0]}
    assert original not in owned["materials"]


def test_material_import_targets_exclude_only_selected_dna_meshes(scene_meshes):
    create, _ = scene_meshes
    points, faces = [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]]
    head, teeth, other = [
        create(name, points, faces)
        for name in (
            "Test_head_lod0_mesh",
            "Test_teeth_lod0_mesh",
            "Other_head_lod0_mesh",
        )
    ]
    component = SimpleNamespace(name="Test", _material_import_exclusions={"head_lod0_mesh"})
    targets = CharacterComponentBase._material_import_targets(
        component, {"head_lod": "head_shader", "teeth_lod": "teeth_shader"}
    )
    assert targets == {"head_lod": [], "teeth_lod": [teeth]}
    assert head not in targets["head_lod"] and other not in targets["head_lod"]
