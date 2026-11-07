"""Direct SDK template preparation must preserve the existing scene import domain."""

import json

import numpy as np
import pytest

from character_dna.bindings import load_converter_core
from character_dna.editors.converter.core import capture_edit_bone_state
from character_dna.editors.converter.native.topology import buffers
from character_dna.editors.converter.utilities import _mesh_positions_array, _mesh_triangles
from constants import TEST_DNA_FOLDER
from fixtures.rigs import RigProfile


@pytest.mark.parametrize("kind", ["head", "body"])
def test_template_matches_imported_scene(rig_factory, tmp_path, kind):
    profile = RigProfile(components=(kind,), lods=(0,), shape_keys=False, face_board=False, materials=False)
    with rig_factory.lease(profile, tmp_path / "scene") as instance:
        _compare(instance, kind)


def _compare(instance, kind):
    import bpy

    before = (set(bpy.data.objects), set(bpy.data.meshes), set(bpy.data.armatures))
    template = load_converter_core().DnaTemplate(str(TEST_DNA_FOLDER / "ada" / f"{kind}.dna"))
    info = json.loads(template.info())
    assert before == (set(bpy.data.objects), set(bpy.data.meshes), set(bpy.data.armatures))
    rig = getattr(instance, f"{kind}_rig")
    states = capture_edit_bone_state(rig)
    for name, state in info["states"].items():
        actual = np.asarray(state["matrix"]).reshape(4, 4)
        expected = np.asarray(states[name][0])
        # Match the established channel reconstruction rotation tolerance;
        # locations remain constrained independently in metres.
        np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=2e-5, rtol=0, err_msg=name)
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=2e-6, rtol=0, err_msg=name)
        assert abs(state["length"] - states[name][1]) < 2e-6
    for record in info["meshes"]:
        obj = bpy.data.objects[f"{instance.name}_{record['name']}"]
        mesh = template.mesh(record["name"])
        np.testing.assert_array_equal(
            np.frombuffer(mesh["positions"], dtype=np.float32).reshape(-1, 3), _mesh_positions_array(obj.data)
        )
        corners, sizes = buffers(obj.data)
        np.testing.assert_array_equal(np.frombuffer(mesh["corners"], dtype=np.int32), corners)
        np.testing.assert_array_equal(np.frombuffer(mesh["sizes"], dtype=np.int32), sizes)
        np.testing.assert_array_equal(
            np.frombuffer(mesh["triangles"], dtype=np.int32).reshape(-1, 3),
            _mesh_triangles(obj.data),
            err_msg=record["name"],
        )
        groups = mesh["groups"]
        names = {group.index: group.name for group in obj.vertex_groups}
        actual = {}
        for vertex in obj.data.vertices:
            for influence in vertex.groups:
                actual.setdefault(names[influence.group], []).append((vertex.index, influence.weight))
        assert groups == actual


def test_preparing_a_lower_lod_preserves_the_lod0_domain():
    from character_dna.editors.converter.native.template import TemplateReader

    reader = TemplateReader(str(TEST_DNA_FOLDER / "ada" / "head.dna"), "head")
    lod0 = reader.getMeshIndicesForLOD(0)
    mesh = reader.prepare_mesh("head_lod1_mesh")
    assert mesh.index not in lod0
    assert reader.getMeshIndicesForLOD(0) == lod0
    assert reader.getMeshName(mesh.index) == mesh.name
    assert reader.prepare_mesh(mesh.name) is mesh
