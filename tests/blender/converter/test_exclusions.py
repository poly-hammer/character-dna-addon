"""DNA-only exclusion regions are inferred without weakening supplied topology."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.dna_io import DNAImporter, mesh_data
from character_dna.editors.converter import correspondence, materials
from character_dna.editors.converter.pipeline import ConversionPipeline, positions

from . import test_mesh_parts


meshes = test_mesh_parts.meshes


@pytest.fixture
def pair(meshes):
    # The missing triangle sits behind a supplied square. IDs deliberately interleave.
    points = np.array(
        [(0, 0, 0), (0.3, 0.3, 0.2), (1, 0, 0), (0.7, 0.3, 0.2), (1, 1, 0), (0.5, 0.6, 0.2), (0, 1, 0)],
        dtype=np.float32,
    )
    faces = [[0, 2, 4], [0, 4, 6], [1, 3, 5]]
    template = meshes("Template", points, faces)
    uv = np.array([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1), (2, 0), (3, 0), (2, 1)], dtype=np.float32)
    template.data.uv_layers.new().data.foreach_set("uv", uv.ravel())
    # Rotation, translation and uniform scale are an independent relative-fit oracle.
    linear = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 2]], dtype=np.float32)
    expected = points @ linear + np.array([3, 4, 5], dtype=np.float32)
    source = meshes("Teeth only", expected[[0, 2, 4, 6]], [[0, 1, 2], [0, 2, 3]])
    source.data.uv_layers.new().data.foreach_set("uv", uv[:6].ravel())
    return source, template, expected


@pytest.mark.parametrize("reordered", [False, True])
def test_copy_preserves_observations_and_fits_only_excluded_vertices(pair, reordered):
    source, target, expected = pair
    if reordered:
        # A full remap forces UV matching, not implicit identity correspondence.
        points = positions(source)[::-1]
        uv = np.empty(len(source.data.loops) * 2, dtype=np.float32)
        source.data.uv_layers.active.data.foreach_get("uv", uv)
        source.data.clear_geometry()
        source.data.from_pydata(points, [], [[3, 2, 1], [3, 1, 0]])
        source.data.uv_layers.new().data.foreach_set("uv", uv)
    props = SimpleNamespace(
        extra_meshes=[SimpleNamespace(role="teeth", fit_method="COPY", scene_object=source, material_source=None)],
        validate_uvs=True,
        uv_tolerance=1e-5,
    )
    pipeline = ConversionPipeline(
        SimpleNamespace(_properties=props, _owned={"materials": set()}),
        bpy.context,
        {
            "meshes": [
                {"id": "teeth", "method": "COPY", "mesh": "teeth_lod0_mesh"},
            ]
        },
    )
    pipeline.objects = {"teeth": target}
    pipeline.exclusions = {"teeth": {"EXCLUDE_Tongue": [1, 3, 5], "Exclude_Overlap": [3]}}
    original = positions(source)
    before = set(bpy.data.meshes)
    pipeline._supplied().apply(1)
    np.testing.assert_array_equal(positions(target)[[0, 2, 4, 6]], expected[[0, 2, 4, 6]])
    np.testing.assert_allclose(positions(target)[[1, 3, 5]], expected[[1, 3, 5]], atol=1e-6)
    np.testing.assert_array_equal(positions(source), original)
    assert len(target.data.vertices) == 7 and len(target.data.polygons) == 3
    assert pipeline.report["meshes"]["teeth"]["mapping"]["relative_vertices"] == 3
    tongue = target.data.materials[target.data.polygons[2].material_index]
    assert tongue.name == "Tongue"
    assert tongue in pipeline.operator._owned["materials"]
    assert target.data.materials[target.data.polygons[0].material_index] is None
    assert set(bpy.data.meshes) == before


def test_complete_input_takes_precedence_over_exclusion_masks(pair):
    _, target, _ = pair
    result, details = correspondence.match_with_exclusions(target.data, target.data, {"EXCLUDE_Tongue": [1, 3, 5]})
    np.testing.assert_array_equal(result, np.arange(7))
    assert "excluded_groups" not in details
    assert details["uv_excluded_groups"] == ["EXCLUDE_Tongue"]
    assert details["uv_ignored_corners"] == 3


def test_complete_input_validates_uvs_without_excluded_corners(pair, meshes):
    _, target, _ = pair
    source = meshes("Complete teeth", positions(target), [list(p.vertices) for p in target.data.polygons])
    uv = np.array([loop.uv[:] for loop in target.data.uv_layers.active.data], dtype=np.float32)
    uv[-3:] += [0.25, -0.4]
    source.data.uv_layers.new().data.foreach_set("uv", uv.ravel())
    with pytest.raises(ValueError, match="UVs differ"):
        correspondence.match(source.data, target.data, require_uvs=True)
    before = set(bpy.data.meshes)
    mapping, details = correspondence.match_with_exclusions(
        source.data, target.data, {"EXCLUDE_Tongue": [1, 3, 5]}, require_uvs=True
    )
    np.testing.assert_array_equal(mapping, np.arange(7))
    assert details["uv_excluded_groups"] == ["EXCLUDE_Tongue"]
    assert details["uv_ignored_corners"] == 3 and details["uv_max_error"] == 0
    assert "relative_vertices" not in details  # Present, mapped anatomy stays copied.
    assert set(bpy.data.meshes) == before
    np.testing.assert_array_equal(np.array([loop.uv[:] for loop in source.data.uv_layers.active.data]), uv)


def test_exclusion_validation_still_rejects_unmasked_uv_errors(pair, meshes):
    _, target, _ = pair
    source = meshes("Complete teeth", positions(target), [list(p.vertices) for p in target.data.polygons])
    uv = np.array([loop.uv[:] for loop in target.data.uv_layers.active.data], dtype=np.float32)
    uv[-3:] += 0.5
    uv[0] += 0.1
    source.data.uv_layers.new().data.foreach_set("uv", uv.ravel())
    with pytest.raises(
        ValueError, match=r"Excluded 3 UV corners using EXCLUDE_Tongue.*UVs differ.*1 of 6 face corners"
    ):
        correspondence.match_with_exclusions(source.data, target.data, {"EXCLUDE_Tongue": [1, 3, 5]}, require_uvs=True)


def test_exclusion_retry_cannot_mask_entire_uv_layout(pair, meshes):
    _, target, _ = pair
    source = meshes("All excluded", positions(target), [list(p.vertices) for p in target.data.polygons])
    uv = np.array([loop.uv[:] for loop in target.data.uv_layers.active.data], dtype=np.float32) + 0.5
    source.data.uv_layers.new().data.foreach_set("uv", uv.ravel())
    with pytest.raises(ValueError, match="cannot remove the entire UV layout"):
        correspondence.match_with_exclusions(
            source.data, target.data, {"EXCLUDE_All": list(range(7))}, require_uvs=True
        )


@pytest.mark.parametrize("material_input", ["none", "same", "independent_full"])
@pytest.mark.parametrize("complete", [False, True])
def test_excluded_region_materials_are_blank_only_when_fallback_is_used(pair, meshes, material_input, complete):
    source, target, _ = pair
    if complete:
        source = meshes("Complete input", positions(target), [list(face.vertices) for face in target.data.polygons])
    material_source = None
    if material_input == "same":
        material_source = source
    elif material_input == "independent_full":
        material_source = meshes(
            "Full material input", positions(target), [list(face.vertices) for face in target.data.polygons]
        )
    original = bpy.data.materials.new("Tongue")
    original.use_nodes = True
    original.node_tree.nodes.new("ShaderNodeTexImage").label = "Existing texture"
    target.data.materials.append(original)
    if material_source is not None:
        material_source.data.materials.append(original)
        if material_input == "independent_full":
            material_source.data.materials.append(bpy.data.materials.new("Other source material"))
            material_source.data.polygons[0].material_index = 1
    props = SimpleNamespace(
        extra_meshes=[
            SimpleNamespace(role="teeth", fit_method="COPY", scene_object=source, material_source=material_source)
        ],
        new_name="BlankRegion",
        validate_uvs=False,
    )
    pipeline = ConversionPipeline(
        SimpleNamespace(_properties=props, _owned={"materials": set()}),
        bpy.context,
        {"meshes": [{"id": "teeth", "method": "COPY", "mesh": "teeth_lod0_mesh"}]},
    )
    pipeline.objects = {"teeth": target}
    pipeline.exclusions = {"teeth": {"EXCLUDE_Tongue": [1, 3, 5]}}
    pipeline._supplied()
    tongue = target.data.materials[target.data.polygons[2].material_index]
    if complete:
        assert tongue == original or tongue.name.startswith("BlankRegion_teeth_shader")
        assert any(node.type == "TEX_IMAGE" for node in tongue.node_tree.nodes)
    else:
        assert tongue != original and tongue.name.startswith("Tongue.")
        assert tongue.use_nodes
        assert {node.type for node in tongue.node_tree.nodes} == {"BSDF_PRINCIPLED", "OUTPUT_MATERIAL"}
        assert tongue in pipeline.operator._owned["materials"]
        assert all(target.data.materials[face.material_index] != tongue for face in target.data.polygons[:2])
    assert original.node_tree.nodes.get("Image Texture").label == "Existing texture"


def test_multiple_exclusions_get_separate_materials_with_deterministic_boundaries(meshes):
    target = meshes(
        "Regions", [(x, y, 0) for y in range(2) for x in range(4)], [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]]
    )
    # Border face has two masked vertices; it belongs to that omitted region too.
    materials.assign_exclusion_materials(
        target,
        {"EXCLUDE_Tongue": [0, 4], "EXCLUDE_Gums": [3, 7]},
        np.array([True, False, False, True, True, False, False, True]),
    )
    assigned = [target.data.materials[face.material_index] for face in target.data.polygons]
    assert assigned[0].name == "Tongue" and assigned[1] is None and assigned[2].name == "Gums"
    assert assigned[0] != assigned[2]


def test_exclusion_material_allocation_failure_still_tracks_new_datablocks(pair, monkeypatch):
    _, target, _ = pair
    pipeline = ConversionPipeline(
        SimpleNamespace(_properties=SimpleNamespace(extra_meshes=[]), _owned={"materials": set()}),
        bpy.context,
        {"meshes": []},
    )
    pipeline.objects = {"teeth": target}
    created = []

    def fail(*_args):
        created.append(bpy.data.materials.new("Failed exclusion material"))
        raise RuntimeError("assignment failed")

    monkeypatch.setattr(materials, "assign_exclusion_materials", fail)
    with pytest.raises(RuntimeError, match="assignment failed"):
        pipeline._assign_exclusion_materials("teeth", np.full(7, -1, dtype=np.int32))
    assert pipeline.operator._owned["materials"] == set(created)


@pytest.mark.parametrize("invalid", ["no_mask", "all_masked", "no_uvs", "wrong_mask", "uv_edits"])
def test_fallback_rejects_unproved_or_incomplete_matches(pair, invalid):
    source, target, _ = pair
    mask = {"EXCLUDE_Tongue": [1, 3, 5]}
    if invalid == "no_mask":
        mask = {}
    if invalid == "all_masked":
        mask = {"EXCLUDE_All": list(range(7))}
    if invalid == "wrong_mask":
        mask = {"EXCLUDE_Wrong": [0, 1, 3]}
    if invalid == "no_uvs":
        source.data.uv_layers.remove(source.data.uv_layers.active)
    if invalid == "uv_edits":
        source.data.uv_layers.active.data[0].uv.x += 0.02
    before = set(bpy.data.meshes)
    with pytest.raises(ValueError):
        correspondence.match_with_exclusions(source.data, target.data, mask, require_uvs=True)
    assert set(bpy.data.meshes) == before


def test_uv_edits_require_explicitly_disabled_uv_validation(pair):
    source, target, _ = pair
    source.data.uv_layers.active.data[0].uv.x += 0.02
    result, details = correspondence.match_with_exclusions(source.data, target.data, {"EXCLUDE_Tongue": [1, 3, 5]})
    assert result.tolist() == [0, -1, 1, -1, 2, -1, 3]
    assert details["uv_max_error"] == pytest.approx(0.02)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_partial_material_assignment_never_uses_missing_vertex_as_source_index(pair, meshes, uniform, independent):
    source, target, _ = pair
    material_source = (
        meshes("Independent materials", positions(source), [list(p.vertices) for p in source.data.polygons])
        if independent
        else source
    )
    original = bpy.data.materials.new("Template tongue")
    target.data.materials.append(original)
    for i in range(1 if uniform else 2):
        material_source.data.materials.append(bpy.data.materials.new(f"Teeth material {i}"))
    if not uniform:
        material_source.data.polygons[1].material_index = 1
    mapping, _ = correspondence.match_with_exclusions(source.data, target.data, {"EXCLUDE_Tongue": [1, 3, 5]})
    if independent:
        pipeline = SimpleNamespace(
            material_sources={"teeth": material_source},
            objects={"teeth": target},
            rows={"teeth": {"mesh": "teeth_lod0_mesh"}},
            operator=SimpleNamespace(_properties=SimpleNamespace(new_name="Exclude"), _owned={"materials": set()}),
        )
        ConversionPipeline._transfer_materials(pipeline, "teeth", source, mapping)
    else:
        materials.transfer(source, target, mapping, instance_name="Exclude", dna_mesh_name="teeth_lod0_mesh")
    for face in target.data.polygons[:2]:
        assert target.data.materials[face.material_index].name.startswith("Exclude_teeth_shader")
    tongue = target.data.materials[target.data.polygons[2].material_index]
    assert tongue != original if uniform else tongue == original


def test_importer_ignores_exclusion_groups_even_if_encoded_as_legacy_joint_weights(meshes):
    source = meshes("Imported", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]])
    reader = SimpleNamespace(
        getVertexLayoutPositionIndices=lambda _: [0, 1, 2],
        getSkinWeightsJointIndices=lambda *_: [0, 1, 2],
        getSkinWeightsValues=lambda *_: [0.7, 1.0, 1.0],
        getJointName=lambda i: ["Jaw", "EXCLUDE_Tongue", "Exclude_Other"][i],
    )
    DNAImporter.set_vertex_groups(SimpleNamespace(_dna_reader=reader), 0, source)
    assert [g.name for g in source.vertex_groups] == ["Jaw"]
    assert source.vertex_groups["Jaw"].weight(0) == pytest.approx(0.7)
    assert list(DNAImporter.get_dna_vertex_groups(SimpleNamespace(_dna_reader=reader), 0)) == [0]


@pytest.mark.parametrize(
    "mask", [{"Other": [0]}, {"EXCLUDE_A": [True]}, {"EXCLUDE_A": [-1]}, {"EXCLUDE_A": [7]}, {"EXCLUDE_A": [1, 1]}, []]
)
def test_malformed_exclusion_metadata_is_rejected(pair, mask):
    _, target, _ = pair
    fingerprint = mesh_data.topology(7, [list(p.vertices) for p in target.data.polygons])
    record = {"name": "test", "topology": fingerprint, "colors": [], "exclusions": mask}
    with pytest.raises(ValueError, match="exclusion"):
        mesh_data.validate_record(record, fingerprint)
