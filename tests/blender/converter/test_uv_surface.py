"""UV correspondence keeps source face identity and template-owned UV layers."""

import json

import bpy
import numpy as np
import pytest

from character_dna.editors.converter import materials, templates
from character_dna.editors.converter.mesh.parts import extract
from character_dna.editors.converter.native.supplied import capture
from character_dna.editors.converter.properties import get_converter_properties


@pytest.fixture
def meshes():
    objects = []
    original_materials = set(bpy.data.materials)

    def create(name, n, offsets):
        points, faces, uv = [], [], []
        for offset in offsets:
            start = len(points)
            points.extend((x / n + offset, y / n, x / n + y / n) for y in range(n + 1) for x in range(n + 1))
            for y in range(n):
                for x in range(n):
                    local = [y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x]
                    faces.append([i + start for i in local])
                    uv.extend((points[i + start][0] - offset, points[i + start][1]) for i in local)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(points, [], faces)
        mesh.uv_layers.new(name="Primary").data.foreach_set("uv", np.asarray(uv).ravel())
        obj = bpy.data.objects.new(name, mesh)
        objects.append(obj)
        return obj

    yield create
    for obj in objects:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
    for material in set(bpy.data.materials) - original_materials:
        bpy.data.materials.remove(material)


def test_side_subset_surface_binding_keeps_original_material_faces_and_extra_uvs(meshes):
    source = meshes("Paired input", 2, [-2, 1])
    target = meshes("DNA target", 4, [1])
    for name in ("Right source", "Left source"):
        source.data.materials.append(bpy.data.materials.new(name))
    for face in source.data.polygons:
        face.material_index = int(face.index >= 4)
    target.data.uv_layers.new(name="HeadOnly")
    target.data.uv_layers["Primary"].active_render = True
    target.data.uv_layers.active_index = 1
    uv_before = [[tuple(loop.uv) for loop in layer.data] for layer in target.data.uv_layers]
    props = {"validate_uvs": True, "uv_tolerance": 0.001}
    with extract(source.data, "LEFT") as (mesh, ids):
        job = capture("eye", source, mesh, ids, target.data, {}, props, "test", correspondence="AUTO")
    job.commit()
    values, indices, observed, report = job.take_result()
    details = json.loads(report)
    binding = materials.binding_from_result(
        np.frombuffer(indices, dtype=np.int32), np.frombuffer(observed, dtype=np.bool_), details
    )
    assert isinstance(binding, materials.SurfaceBinding)
    assert np.all((binding.faces >= 4) & (binding.faces < 8))
    np.testing.assert_allclose(
        np.frombuffer(values, dtype=np.float32).reshape(-1, 3), [v.co[:] for v in target.data.vertices], atol=1e-6
    )
    materials.transfer(source, target, binding, instance_name="UVTest", dna_mesh_name="eyeLeft_lod0_mesh")
    assert [material.name for material in target.data.materials] == ["UVTest_eyeLeft_shader"]
    assert target.data.materials[0] != source.data.materials[1]
    assert all(face.material_index == 0 for face in target.data.polygons)
    assert [[tuple(loop.uv) for loop in layer.data] for layer in target.data.uv_layers] == uv_before
    assert len(source.data.vertices) == 18


def test_scan_store_mesh_defaults_and_explicit_settings():
    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("UV surface defaults")
    bpy.context.window.scene = scene
    try:
        props = get_converter_properties()
        props.template = str(templates.TEMPLATE_FOLDER / "3d_scan_store_full_body.json")
        eye = next(item for item in props.extra_meshes if item.role == "eye_left")
        teeth = next(item for item in props.extra_meshes if item.role == "teeth")
        rows = {row["id"]: row for row in templates.from_properties(props)["meshes"]}
        assert rows["eye_left"]["correspondence"] == "AUTO"
        assert rows["teeth"]["uv_tolerance"] == pytest.approx(0.0015)
        eye.correspondence = "EXACT"
        teeth.uv_tolerance = 0.002
        rows = {row["id"]: row for row in templates.from_properties(props)["meshes"]}
        assert rows["eye_left"]["correspondence"] == "EXACT"
        assert rows["teeth"]["uv_tolerance"] == pytest.approx(0.002)
        templates.apply_template(props, templates.default_template())
        assert all(item.correspondence == "EXACT" for item in props.extra_meshes)
        assert all(item.uv_tolerance == pytest.approx(0.001) for item in props.extra_meshes)
    finally:
        bpy.context.window.scene = previous
        bpy.data.scenes.remove(scene)
