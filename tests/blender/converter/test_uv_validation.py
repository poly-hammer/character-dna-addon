"""UV layout validation keeps face corners, seams and the primary DNA map distinct."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.components.base import CharacterComponentBase
from character_dna.editors.converter import correspondence, solver, templates
from character_dna.editors.converter.operators import CHARACTER_DNA_OT_convert_to_dna


@pytest.fixture
def meshes():
    created = []

    def make(faces, uv):
        count = max(v for face in faces for v in face) + 1
        mesh = bpy.data.meshes.new("UV layout test")
        mesh.from_pydata([(v, 0, 0) for v in range(count)], [], faces)
        if uv is not None:
            mesh.uv_layers.new(name="DiffuseUV").data.foreach_set("uv", np.asarray(uv, dtype=np.float32).ravel())
        created.append(mesh)
        return mesh

    yield make
    for mesh in created:
        bpy.data.meshes.remove(mesh)


FACES = [[0, 1, 4, 3], [1, 2, 5, 4]]
# The shared vertices have different UVs on either side of an island seam.
UV = np.array([[(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)]], dtype=float)


def reordered(meshes, uv):
    permutation = np.array([5, 3, 1, 4, 2, 0])
    faces = [np.roll(permutation[FACES[i]], 1).tolist() for i in (1, 0)]
    return meshes(faces, np.roll(uv[::-1], 1, axis=1)), permutation


def test_reordered_vertices_faces_corners_and_seams(meshes):
    target = meshes(FACES, UV)
    source, permutation = reordered(meshes, UV)
    mapping, details = correspondence.match(source, target, require_uvs=True)
    np.testing.assert_array_equal(mapping, permutation)
    assert details["uv_mismatched_corners"] == details["uv_max_error"] == 0


def test_near_identical_uvs_across_rounding_boundaries(meshes):
    uv = UV + 0.0000049
    target = meshes(FACES, uv)
    source, permutation = reordered(meshes, uv + 0.0000003)
    mapping, details = correspondence.match(source, target, tolerance=1e-5, require_uvs=True)
    np.testing.assert_array_equal(mapping, permutation)
    assert 0 < details["uv_max_error"] < 1e-5


@pytest.mark.parametrize("primary_matches", [True, False])
def test_active_secondary_maps_never_override_primary(meshes, primary_matches):
    target = meshes(FACES, UV)
    source = meshes(FACES, UV if primary_matches else UV + 0.2)
    before = solver.topology_digest(target)
    for mesh in (source, target):
        extra = mesh.uv_layers.new(name="HeadOnly")
        extra.data.foreach_set("uv", (UV + (0.7 if mesh == target and primary_matches else 0)).ravel())
        mesh.uv_layers.active = extra
        extra.active_render = True
    assert solver.topology_digest(target) == before
    if primary_matches:
        _, details = correspondence.match(source, target, require_uvs=True)
        assert details["input_uv_map"] == details["template_uv_map"] == "DiffuseUV"
    else:
        with pytest.raises(ValueError, match=r"UVs differ.*8 of 8 face corners"):
            correspondence.match(source, target, require_uvs=True)


def test_seam_corners_are_not_averaged(meshes):
    changed = UV.copy()
    changed[0, 1, 0] += 0.1
    changed[1, 0, 0] -= 0.1
    with pytest.raises(ValueError, match="2 of 8 face corners"):
        correspondence.match(meshes(FACES, changed), meshes(FACES, UV), require_uvs=True)


@pytest.mark.parametrize("change", [np.array([1, 0]), np.array([0.008, 0.008])])
def test_tiles_and_diagonal_distance_are_not_silently_accepted(meshes, change):
    with pytest.raises(ValueError, match="UVs differ"):
        correspondence.match(meshes(FACES, UV + change), meshes(FACES, UV), tolerance=0.01, require_uvs=True)


def test_swapped_islands_fail_even_with_the_same_unordered_uvs(meshes):
    with pytest.raises(ValueError, match="UVs differ"):
        correspondence.match(meshes(FACES, UV[::-1]), meshes(FACES, UV), require_uvs=True)


def test_missing_primary_requires_explicit_validation_opt_out(meshes):
    source, target = meshes(FACES, None), meshes(FACES, UV)
    with pytest.raises(ValueError, match="requires a primary UV map"):
        correspondence.match(source, target, require_uvs=True)
    mapping, details = correspondence.match(source, target, require_uvs=False)
    np.testing.assert_array_equal(mapping, np.arange(6))
    assert details["uv_max_error"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_uvs_are_rejected(meshes, bad):
    changed = UV.copy()
    changed[0, 0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        correspondence.match(meshes(FACES, changed), meshes(FACES, UV), require_uvs=True)


def test_full_mesh_uv_error_cannot_hide_behind_an_exclusion(meshes):
    with pytest.raises(ValueError, match="UVs differ"):
        correspondence.match_with_exclusions(
            meshes(FACES, UV + 0.1), meshes(FACES, UV), {"EXCLUDE_Test": [0, 3]}, require_uvs=True
        )


@pytest.mark.parametrize("preserve", [True, False])
def test_importer_shading_opt_out_and_legacy_default(monkeypatch, preserve):
    from character_dna import utilities
    from character_dna.constants import BODY_MESH_SHADER_MAPPING

    calls = []
    monkeypatch.setattr(utilities, "set_viewport_shading", calls.append)
    component = SimpleNamespace(
        name="Shading test",
        component_type="body",
        dna_import_properties=SimpleNamespace(preserve_viewport_shading=True, import_materials=True)
        if preserve
        else None,
        _material_import_targets=lambda mapping: {key: [] for key in mapping},
        _material_import_exclusions=set(BODY_MESH_SHADER_MAPPING),
        _purge_existing_materials=lambda *_: None,
        _load_texture_maps=False,
    )
    CharacterComponentBase.import_materials(component)
    assert calls == ([] if preserve else ["MATERIAL"])
    rna = bpy.ops.character_dna.import_dna.get_rna_type().properties["preserve_viewport_shading"]
    assert rna.is_hidden and not rna.default


def test_converter_always_requests_preserved_shading():
    flags = SimpleNamespace()
    CHARACTER_DNA_OT_convert_to_dna._apply_import_properties(SimpleNamespace(properties=flags))
    assert flags.preserve_viewport_shading


@pytest.mark.parametrize("path", templates.BUNDLED_TEMPLATES)
def test_all_bundled_templates_enable_uv_validation(path):
    import json

    assert all(row["validate_uvs"] for row in json.loads(path.read_text())["meshes"])
