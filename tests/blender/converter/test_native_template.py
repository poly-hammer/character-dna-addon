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


def test_scan_store_head_identity_pose_preserves_vertices_with_raw_dna_weights():
    """The male01 template's non-unit weights must not scale the face."""
    from character_dna.editors.shared import topology_profiles

    native = load_converter_core()
    template = native.DnaTemplate(str(topology_profiles.ROOT.parent / "dna/3d_scan_store/full_body/head.dna"))
    info = json.loads(template.info())
    mesh = template.mesh("head_lod0_mesh")
    names = info["names"]
    points = np.frombuffer(mesh["positions"], np.float32).reshape(-1, 3).astype(float)
    weights = np.frombuffer(template.weights("head_lod0_mesh", names), np.float64).reshape(-1, len(names))
    # Keep this fixture meaningful: it contains the raw sums that exposed the bug.
    assert np.count_nonzero(np.abs(weights.sum(axis=1) - 1.0) > 1e-5) > 0
    parents = [names[i] if i >= 0 else None for i in info["parents"]]
    edges = np.frombuffer(mesh["edges"], np.int32).reshape(-1, 2)
    triangles = np.frombuffer(mesh["triangles"], np.int32).reshape(-1, 3)
    identity = np.eye(4).reshape(1, 16)
    values, report = native.pose_accept(
        [names[0]],
        identity,
        identity,
        points,
        [("head", points, weights, names, parents, edges, triangles, True)],
        [],
    )
    np.testing.assert_array_equal(np.frombuffer(values["head"], np.float64).reshape(-1, 3), points)
    assert json.loads(report)["head"]["new_contact_pairs"] == 0


def test_template_fingerprint_respects_metadata_without_a_primary_uv(monkeypatch):
    from types import SimpleNamespace

    from character_dna.editors.converter.native import direct
    from character_dna.editors.converter.native.template import TemplateReader

    reader = TemplateReader(str(TEST_DNA_FOLDER / "ada" / "body.dna"), "body")
    mesh = reader.meshes["body_lod0_mesh"]
    digest = load_converter_core().topology_digest(mesh.corners, mesh.sizes, len(mesh.positions), None)
    pipeline = object.__new__(direct.DirectPipeline)
    pipeline.operator = SimpleNamespace(
        _component_types=["body"],
        _properties=SimpleNamespace(dna_files={"body": "retained"}),
        _components={},
        _dna_mesh_names={},
    )
    pipeline.rows = {"body": {"component": "body", "mesh": mesh.name, "topology_sha256": digest}}
    pipeline.mesh_metadata = {"body": {mesh.index: {"uv_layers": []}}}
    pipeline.objects, pipeline.base, pipeline.position_ids = {}, {}, {}
    monkeypatch.setattr(direct, "TemplateReader", lambda *_args: reader)
    for method in ("_resolve_profiles", "_prepare_assembly", "_load_masks", "_record"):
        monkeypatch.setattr(pipeline, method, lambda *_args: None)
    pipeline._import()
    assert pipeline.prepared["body"].uv is None
    assert pipeline.prepared["body"].primary_uv is not None
