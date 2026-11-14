"""Exact correspondence, sparse DNA domains and bundled topology fitting data."""

import json

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.dna_io import get_dna_reader, merged_graph, mesh_data, release_dna_handle
from character_dna.dna_io.vertex_indices import scene_to_dna
from character_dna.editors.converter import assembly, correspondence, templates
from character_dna.editors.converter.lifecycle import publish_files
from character_dna.editors.shared import bone_mapping, topology_profiles


@pytest.fixture
def meshes():
    created = []

    def make(name, points, faces, uv=None):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(points, [], faces)
        if uv is not None:
            mesh.uv_layers.new().data.foreach_set("uv", np.asarray(uv, dtype=np.float32).ravel())
        created.append(mesh)
        return mesh

    yield make
    for mesh in created:
        bpy.data.meshes.remove(mesh, do_unlink=True)


def test_reordered_input_with_locally_changed_uvs(meshes):
    points = [(x, y, 0) for y in range(2) for x in range(4)]
    faces = [[i, i + 1, i + 5, i + 4] for i in range(3)]
    uv = np.array([points[v][:2] for face in faces for v in face], dtype=float).reshape(3, 4, 2)
    target = meshes("template", points, faces, uv)
    permutation = np.array([4, 7, 2, 0, 1, 6, 5, 3])
    transformed = np.empty((8, 3))
    transformed[permutation] = np.asarray(points) * 1.2 + [0.1, 0.2, 0.3]
    order = [2, 0, 1]
    source_uv = uv[order].copy()
    source_uv[0, 1] += 0.001
    source = meshes("input", transformed, [permutation[faces[i]].tolist() for i in order], source_uv)
    mapping, report = correspondence.match(source, target)
    np.testing.assert_array_equal(mapping, permutation)
    assert report["uv_max_error"] > 0.001
    with pytest.raises(ValueError, match="UVs differ"):
        correspondence.match(source, target, require_uvs=True)


@pytest.mark.parametrize("reversed_winding", [False, True])
def test_reject_wrong_connectivity_or_winding(meshes, reversed_winding):
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    target = meshes("template", points, [[0, 1, 2, 3]], [(0, 0), (1, 0), (1, 1), (0, 1)])
    face = [3, 2, 1, 0] if reversed_winding else [0, 2, 1, 3]
    source = meshes("invalid", points, [face], [points[v][:2] for v in face])
    with pytest.raises(ValueError):
        correspondence.match(source, target)


def test_ambiguous_uv_islands_do_not_guess(meshes):
    points = [(i, 0, 0) for i in range(8)]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)] * 2
    target = meshes("template", points, faces, uv)
    source = meshes("ambiguous", points, faces[::-1], np.roll(np.array(uv).reshape(2, 4, 2), 1, axis=1))
    with pytest.raises(ValueError, match="Ambiguous"):
        correspondence.match(source, target)


def test_sparse_position_map_preserves_holes(meshes):
    mesh = meshes("compact", [(0, 0, 0)] * 4, [[0, 1, 2, 3]])
    assert scene_to_dna(mesh, 9, [[1, 3, 5, 8]]) == [1, 3, 5, 8]
    with pytest.raises(ValueError, match="face topology"):
        scene_to_dna(mesh, 9, [[1, 5, 3, 8]])


def test_native_weld_keeps_skin_weights_in_the_position_domain():
    def source(count, joints):
        return {
            "positions": {axis: [0.0] * count for axis in ("xs", "ys", "zs")},
            "skinWeights": [{"weights": [1.0], "jointIndices": [joint]} for joint in joints],
            "maximumInfluencePerVertex": 1,
            "normals": {axis: [] for axis in ("xs", "ys", "zs")},
            "textureCoordinates": {axis: [] for axis in ("us", "vs")},
            "layouts": {domain: [] for domain in ("positions", "textureCoordinates", "normals")},
            "faces": [],
            "blendShapeTargets": [],
        }

    result, maps = merged_graph._weld(source(3, [8, 9, 10]), source(5, [1, 2, 3]), {0: 2})
    assert len(result["skinWeights"]) == 7
    assert result["skinWeights"][4] == {"weights": [], "jointIndices": []}
    assert result["skinWeights"][maps["head"][1]]["jointIndices"] == [9]
    assert result["skinWeights"][maps["head"][2]]["jointIndices"] == [10]
    assert result["skinWeights"][maps["head"][0]]["jointIndices"] == [3]


@pytest.fixture
def scan_readers():
    files = templates.resolve_dna_files(
        json.loads((templates.TEMPLATE_FOLDER / "3d_scan_store_full_body.json").read_text()),
        str(templates.TEMPLATE_FOLDER / "3d_scan_store_full_body.json"),
    )
    readers = {kind: get_dna_reader(path) for kind, path in files.items()}
    yield readers
    for reader in readers.values():
        release_dna_handle(reader)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_transferred_anchors_preserve_neutral_and_translate(scan_readers, kind):
    reader = scan_readers[kind]
    profile = topology_profiles.resolve(reader, kind)
    assert profile == "3d_scan_store/full_body"
    vertices = {}
    count = 0
    for index in reader.getMeshIndicesForLOD(0):
        name = reader.getMeshName(index)
        topology_profiles.validate_mesh(profile, name, reader, index)
        points = np.column_stack([getattr(reader, f"getVertexPosition{axis}s")(index) for axis in "XYZ"])
        vertices[name] = points.tolist()
        used = sorted({v for face in mesh_data.reader_faces(reader, index) for v in face})
        for category in ("vertex_bone_mapping", "centroid_bone_mapping"):
            for entry in topology_profiles.mappings(profile, category, name).values():
                full = topology_profiles.samples(entry, points)
                compact = topology_profiles.samples(entry, points[used], used)
                np.testing.assert_array_equal(full, compact)
                np.testing.assert_allclose(
                    topology_profiles.samples(entry, points + np.array([1, 2, 3])) - full,
                    np.tile([1, 2, 3], (len(full), 1)),
                    atol=1e-9,
                )
                count += 1
    assert count > 100
    baseline, mask = bone_mapping.fit_neutral_joints_to_mesh(reader, vertices)
    shifted, other = bone_mapping.fit_neutral_joints_to_mesh(
        reader, {name: (np.array(points) + np.array([1, 2, 3])).tolist() for name, points in vertices.items()}
    )
    assert mask == other and sum(mask) > 100
    for before, after, fitted in zip(baseline, shifted, mask, strict=True):
        np.testing.assert_allclose(
            np.asarray(after.translation) - before.translation, [1, 2, 3] if fitted else [0, 0, 0], atol=2e-5
        )


def test_assembly_uses_compact_domains_and_shared_seam(meshes, scan_readers):
    objects = {}
    try:
        for kind, reader in scan_readers.items():
            faces = mesh_data.reader_faces(reader, 0)
            used = sorted({v for face in faces for v in face})
            inverse = {v: i for i, v in enumerate(used)}
            points = np.column_stack([getattr(reader, f"getVertexPosition{axis}s")(0) for axis in "XYZ"]) * 0.01
            mesh = meshes(kind, points[used], [[inverse[v] for v in f] for f in faces])
            obj = bpy.data.objects.new(kind, mesh)
            objects[kind] = obj
            group = obj.vertex_groups.new(name=mesh_data.SEAM_GROUP)
            group.add([inverse[v] for v in mesh_data.record_for(reader, kind, 0)["neckline"]], 1, "REPLACE")
        domain = assembly.prepare(objects, scan_readers, {kind: f"{kind}_lod0_mesh" for kind in objects})
        try:
            assert len(domain.mesh.vertices) == 32078
            assert len(domain.mesh.polygons) == 32076
            assert len(domain.seam) == 110
            for h, b in domain.seam.items():
                assert domain.component_to_assembly["head"][h] == domain.component_to_assembly["body"][b]
        finally:
            domain.close()
    finally:
        for obj in objects.values():
            bpy.data.objects.remove(obj)


def test_scene_binding_failure_rolls_back_published_files(tmp_path):
    staging, destination = tmp_path / "staging", tmp_path / "output"
    staging.mkdir()
    destination.mkdir()
    (destination / "head.dna").write_bytes(b"previous")
    (staging / "head.dna").write_bytes(b"replacement")
    (staging / "body.dna").write_bytes(b"new")

    def fail():
        raise ValueError("scene binding failed")

    with pytest.raises(ValueError, match="scene binding failed"):
        publish_files(staging, destination, fail)
    assert (destination / "head.dna").read_bytes() == b"previous"
    assert not (destination / "body.dna").exists()


def test_joined_input_ignores_stale_component_pointers():
    rows = [SimpleNamespace(role=role, fit_method="COPY") for role in ("head", "body", "teeth")]
    props = SimpleNamespace(input_layout="JOINED", extra_meshes=rows)
    assert [item.role for item in templates.input_rows(props)] == ["teeth"]
