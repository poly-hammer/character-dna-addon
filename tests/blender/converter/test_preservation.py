"""Preserved regions retain depth, fixed boundaries, and template-independent fits."""

import numpy as np
import pytest

from character_dna.editors.converter import preservation, preserve_clearance, templates
from character_dna.editors.converter.properties import get_converter_properties


def pocket():
    """A deep pocket with several exterior rings and a flattened copy target."""
    points, weights, edges = [], [], []
    segments, rings = 24, 12
    for ring in range(rings):
        depth = max(0, ring - 5) * 0.006
        radius = 0.02 + max(0, 5 - ring) * 0.003
        for j in range(segments):
            angle = j * 2 * np.pi / segments
            points.append((radius * np.cos(angle), depth, radius * np.sin(angle)))
            weights.append(min(1, max(0, ring - 5) / 4))
            edges.append((ring * segments + j, ring * segments + (j + 1) % segments))
            if ring:
                edges.append(((ring - 1) * segments + j, ring * segments + j))
    base = np.array(points)
    copied = base.copy()
    copied[:, 1] *= 0.12
    return base, copied.astype(np.float32), np.array(edges), np.array(weights)


def test_preservation_recovers_depth_without_changing_unpainted_vertices():
    base, copied, edges, weights = pocket()
    fitted, report = preservation.fit(base, copied, edges, {"PRESERVE_Pocket": weights})
    np.testing.assert_array_equal(fitted[weights == 0], copied[weights == 0])
    interior = weights == 1
    assert np.mean(fitted[interior, 1]) > np.mean(base[interior, 1]) * 0.85
    assert np.mean(abs(fitted[interior, 1] - base[interior, 1])) < 0.003
    assert np.isfinite(fitted).all() and report["vertices"] == np.count_nonzero(weights)
    assert fitted.dtype == copied.dtype


def test_preservation_strength_zero_and_missing_groups_are_exact_noops():
    base, copied, edges, weights = pocket()
    for masks, strength in (({}, 1), ({"PRESERVE_Pocket": weights}, 0)):
        actual, report = preservation.fit(base, copied, edges, masks, strength)
        np.testing.assert_array_equal(actual, copied)
        assert not report
    half, _ = preservation.fit(base, copied, edges, {"PRESERVE_Pocket": weights}, 0.5)
    full, _ = preservation.fit(base, copied, edges, {"PRESERVE_Pocket": weights}, 1)
    assert np.mean(copied[weights == 1, 1]) < np.mean(half[weights == 1, 1]) < np.mean(full[weights == 1, 1])


def test_preservation_follows_rotation_scale_translation_and_identity():
    base, copied, edges, weights = pocket()
    angle = 0.7
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
    masks = {"PRESERVE_Pocket": weights}
    transformed = base @ rotation * 1.4 + (0.1, 0.2, 1.5)
    actual, _ = preservation.fit(base, transformed, edges, masks)
    np.testing.assert_allclose(actual, transformed, atol=1e-9)
    fitted, _ = preservation.fit(base, copied, edges, masks)
    actual, _ = preservation.fit(base, copied @ rotation * 1.4 + (0.1, 0.2, 1.5), edges, masks)
    np.testing.assert_allclose(actual, fitted @ rotation * 1.4 + (0.1, 0.2, 1.5), atol=1e-7)


def test_overlapping_masks_do_not_double_deform_and_disconnected_regions_are_independent():
    base, copied, edges, weights = pocket()
    expected, _ = preservation.fit(base, copied, edges, {"PRESERVE_One": weights})
    actual, _ = preservation.fit(base, copied, edges, {"PRESERVE_One": weights, "PRESERVE_Two": weights * 0.5})
    np.testing.assert_array_equal(actual, expected)
    offset = np.array((1, 0, 0))
    actual, report = preservation.fit(
        np.concatenate((base, base + offset)),
        np.concatenate((copied, copied + offset)),
        np.concatenate((edges, edges + len(base))),
        {"PRESERVE_Two": np.tile(weights, 2)},
    )
    np.testing.assert_allclose(actual[: len(base)], expected, atol=1e-8)
    np.testing.assert_allclose(actual[len(base) :], expected + offset, atol=1e-8)
    assert len(report["regions"]) == 2


def test_preservation_never_uses_missing_exclusion_vertices_as_copy_targets():
    base, copied, edges, weights = pocket()
    observed = np.ones(len(base), dtype=bool)
    observed[-24:] = False
    masks = {"PRESERVE_Pocket": weights}
    altered = copied.copy()
    altered[~observed] = 50
    expected, _ = preservation.fit(base, copied, edges, masks, observed=observed)
    actual, _ = preservation.fit(base, altered, edges, masks, observed=observed)
    np.testing.assert_array_equal(actual[observed], expected[observed])
    np.testing.assert_array_equal(actual[~observed], altered[~observed])


def test_preservation_without_exterior_anchors_reports_an_actionable_error():
    base, copied, edges, weights = pocket()
    with pytest.raises(ValueError, match="surrounding, unpainted"):
        preservation.fit(base, copied, edges, {"PRESERVE_All": np.ones_like(weights)})


def test_clearance_moves_crossing_faces_even_when_their_vertices_are_outside():
    reference = np.array([(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    reference_triangles = np.array([triangle for a, b, c, d in quads for triangle in ((a, b, c), (a, c, d))])
    copied = np.array([(-0.8, 0, -0.8), (0.8, 0, -0.8), (0.8, 0, 0.8), (-0.8, 0, 0.8), (2, 2, 2)])
    triangles = np.array(((0, 1, 2), (0, 2, 3)))
    movable = np.array((True, True, True, True, False))
    result, report = preserve_clearance.fit(copied, triangles, movable, reference, reference_triangles)
    assert report["intersecting_faces_before"] == 2
    assert report["intersecting_faces_after"] == 0
    np.testing.assert_array_equal(result[~movable], copied[~movable])
    zero, details = preserve_clearance.fit(copied, triangles, movable, reference, reference_triangles, 0)
    np.testing.assert_array_equal(zero, copied)
    assert not details
    half, _ = preserve_clearance.fit(copied, triangles, movable, reference, reference_triangles, 0.5)
    np.testing.assert_allclose(half, (copied + result) / 2)
    clear = copied + np.array((0, 2, 0))
    unchanged, details = preserve_clearance.fit(clear, triangles, movable, reference, reference_triangles)
    np.testing.assert_array_equal(unchanged, clear)
    assert details["corrected_vertices"] == 0


def test_surface_contact_guard_reduces_only_new_crossings():
    # A movable triangle crosses a fixed vertical triangle after preservation.
    copied = np.array(((-2, -1, -1), (-2, 1, -1), (-2, 0, 1), (0, -2, 0), (0, 2, 0), (2, 0, 0)), dtype=float)
    triangles = np.array(((0, 1, 2), (3, 4, 5)))
    movable = np.array((True, True, True, False, False, False))
    candidate = copied.copy()
    candidate[:3, 0] = 1
    result, report = preserve_clearance.limit_contacts(copied, candidate, triangles, movable)
    assert report["new_intersecting_pairs"] == 0
    assert report["limited_vertices"] == 3
    assert np.all(result[:3, 0] < 0)
    np.testing.assert_array_equal(result[~movable], copied[~movable])
    # A crossing already in the copy input is not silently repaired.
    unchanged, report = preserve_clearance.limit_contacts(candidate, candidate, triangles, movable)
    np.testing.assert_array_equal(unchanged, candidate)
    assert report["copied_intersecting_pairs"] == 1
    assert report["limited_vertices"] == 0


def test_old_scan_store_scene_gets_clearance_defaults_without_losing_bindings():
    import json

    props = get_converter_properties()
    path = templates.BUNDLED_TEMPLATES[-1]
    recipe = json.loads(path.read_text())
    recipe["meshes"][0].pop("preserve_clearance_references")
    try:
        templates.apply_template(props, recipe, str(path))
        for item in props.extra_meshes:
            item.property_unset("preserve_shape")
        previous = [(item.role, item.scene_object, item.material_source) for item in props.extra_meshes]
        templates.migrate(props)
        actual = templates.from_properties(props)
        assert actual["meshes"][0]["preserve_clearance_references"] == ["teeth"]
        assert previous == [(item.role, item.scene_object, item.material_source) for item in props.extra_meshes]
    finally:
        templates.apply_template(props, templates.default_template())


def test_clearance_dependencies_reject_cycles_and_relative_references():
    recipe = templates.default_template()
    recipe["meshes"][0]["preserve_clearance_references"] = ["teeth"]
    with pytest.raises(ValueError, match="Copy to Target"):
        templates.validate_template(recipe)
    recipe = templates.default_template("exact")
    recipe["meshes"][0]["preserve_clearance_references"] = ["body"]
    recipe["meshes"][1]["preserve_clearance_references"] = ["head"]
    with pytest.raises(ValueError, match="Cyclic"):
        templates.validate_template(recipe)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "1"])
def test_template_rejects_invalid_preservation_strength(value):
    recipe = templates.default_template()
    recipe["meshes"][0]["preserve_shape"] = value
    with pytest.raises(ValueError, match="Preserve Shape"):
        templates.validate_template(recipe)


def test_preservation_slider_roundtrips_in_template_configuration():
    props = get_converter_properties()
    recipe = templates.default_template()
    recipe["meshes"][0]["preserve_shape"] = 0.35
    try:
        templates.apply_template(props, recipe)
        assert props.extra_meshes[0].preserve_shape == pytest.approx(0.35)
        result = templates.from_properties(props)
        assert result["meshes"][0]["preserve_shape"] == pytest.approx(0.35)
    finally:
        templates.apply_template(props, templates.default_template())
