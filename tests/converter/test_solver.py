"""Numerical invariants and lifecycle regressions for the geometric converter."""

import json

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.editors.converter import templates
from character_dna.editors.converter.lifecycle import ConversionLifecycle, publish_files
from character_dna.editors.converter.pipeline import Transition, surface_transport
from character_dna.editors.converter.properties import get_converter_properties
from character_dna.editors.converter.solver import correspondence, dependency_layers, similarity


def mesh(name, order=(0, 1, 2, 3)):
    data = bpy.data.meshes.new(name)
    points = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    inverse = np.argsort(order)
    data.from_pydata(points[list(order)], [], [inverse.tolist()])
    uv = data.uv_layers.new()
    uv.data.foreach_set("uv", points[:, :2].ravel())
    return data


def test_exact_correspondence_accepts_reordered_vertices_rejects_ambiguous_uvs():
    source, target = mesh("input", (2, 0, 3, 1)), mesh("base")
    try:
        assert correspondence(source, target).tolist() == [1, 3, 0, 2]
        source.uv_layers.active.data[0].uv.x += 0.1
        with pytest.raises(ValueError, match="UVs"):
            correspondence(source, target)
    finally:
        bpy.data.meshes.remove(source)
        bpy.data.meshes.remove(target)


def test_similarity_recovers_rotation_scale_translation_without_reflection():
    rng = np.random.default_rng(52)
    source = rng.normal(size=(50, 3))
    rotation = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    target = source @ rotation * 1.5 + (3, 2, 4)
    linear, offset, rms = similarity(source, target)
    np.testing.assert_allclose(source @ linear + offset, target, atol=1e-12)
    assert rms < 1e-12 and np.linalg.det(linear) > 0
    with pytest.raises(ValueError, match="collinear"):
        similarity(np.zeros((3, 3)), np.ones((3, 3)))


def test_surface_transport_scales_thickness_and_preserves_identity():
    source = np.array(((0, 0, 0), (2, 0, 0), (0, 2, 0)), dtype=float)
    dependent = np.array(((0.5, 0.5, 0.5),))
    actual, _ = surface_transport(source, source * 2 + (1, 2, 3), np.array(((0, 1, 2),)), dependent)
    np.testing.assert_allclose(actual, dependent * 2 + (1, 2, 3), atol=1e-7)
    same, _ = surface_transport(source, source, np.array(((0, 1, 2),)), dependent)
    np.testing.assert_allclose(same, dependent, atol=1e-7)


def test_dependencies_are_order_independent_and_invalid_graphs_fail():
    assert dependency_layers({"b": ["a"], "a": [], "c": ["b"]}) == [["a"], ["b"], ["c"]]
    with pytest.raises(ValueError, match="Cyclic"):
        dependency_layers({"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError, match="Unknown"):
        dependency_layers({"a": ["missing"]})


def test_template_round_trip_keeps_roles_references_and_custom_names():
    props = get_converter_properties()
    recipe = templates.default_template()
    for row in recipe["meshes"]:
        row["mesh"] = "scan_" + row["mesh"]
    templates.apply_template(props, recipe)
    props.extra_meshes.move(0, 7)
    restored = templates.from_properties(props)
    by_id = {row["id"]: row for row in restored["meshes"]}
    assert by_id["saliva"]["references"] == ["teeth"]
    assert by_id["eye_shell"]["references"] == ["head", "eye_left", "eye_right"]
    assert by_id["head"]["mesh"] == "scan_head_lod0_mesh"
    assert templates.validate_template(json.loads(json.dumps(restored))) == restored
    templates.apply_template(props, templates.default_template())


def test_template_rejects_unsafe_resource_aliases_and_missing_clearance_dependencies():
    recipe = templates.default_template()
    recipe["meshes"][0]["mapping"] = "../outside"
    with pytest.raises(ValueError, match="Unknown joint map"):
        templates.validate_template(recipe)
    recipe = templates.default_template()
    recipe["meshes"][0]["clearance_references"] = ["eye_left"]
    with pytest.raises(ValueError, match="Clearance references"):
        templates.validate_template(recipe)


def test_old_skin_rows_migrate_without_reusing_the_enum_number():
    props = get_converter_properties()
    props.extra_meshes.clear()
    props.template_data = ""
    props.relationships_version = 0
    item = props.extra_meshes.add()
    item.dna_mesh_name = "eyeLeft_lod0_mesh"
    item["fit_method"] = 1
    templates.migrate(props)
    assert item.fit_method == "RELATIVE"
    assert item.profile == "RIGID"
    assert [ref.key for ref in item.references if ref.enabled] == ["head"]
    templates.apply_template(props, templates.default_template())


def test_transition_is_linear_and_does_not_advance_the_next_solve():
    data = mesh("transition")
    obj = bpy.data.objects.new("transition", data)
    bpy.context.scene.collection.objects.link(obj)
    transition = Transition("surface")
    transition.add_mesh(obj, np.array([v.co[:] for v in data.vertices]) + 2)
    stages_run = []

    def stages():
        stages_run.append("surface")
        yield transition
        stages_run.append("bones")

    driver = stages()
    try:
        assert next(driver).duration == 1.0
        transition.apply(0.5)
        assert tuple(data.vertices[0].co) == (1, 1, 1)
        assert stages_run == ["surface"]
        transition.commit()
        assert tuple(data.vertices[0].co) == (2, 2, 2)
        assert next(driver, None) is None
        assert stages_run == ["surface", "bones"]
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(data)


def test_escape_cancels_without_export_or_stage_advance():
    calls = []
    fake = SimpleNamespace(cancel=lambda _context: calls.append("cancel"))
    assert ConversionLifecycle.modal(fake, None, SimpleNamespace(type="ESC")) == {"CANCELLED"}
    assert calls == ["cancel"]


def test_timer_events_without_a_timer_attribute_advance_one_stage():
    calls = []
    fake = SimpleNamespace(_last_tick=0.0, _transition=None, _advance=lambda _context: calls.append("advance") or True)
    context = SimpleNamespace(area=None)
    assert ConversionLifecycle.modal(fake, context, SimpleNamespace(type="TIMER")) == {"RUNNING_MODAL"}
    assert calls == ["advance"]


def test_output_transaction_restores_existing_files_on_failure(tmp_path, monkeypatch):
    from pathlib import Path

    staging, destination = tmp_path / "stage", tmp_path / "out"
    staging.mkdir()
    destination.mkdir()
    for name in ("head.dna", "body.dna"):
        (staging / name).write_text("new")
        (destination / name).write_text("old")
    replace = Path.replace

    def fail_body(source, target):
        if source == staging / "head.dna":
            raise OSError("injected publication failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_body)
    with pytest.raises(OSError, match="injected"):
        publish_files(staging, destination)
    assert (destination / "head.dna").read_text() == "old"
    assert (destination / "body.dna").read_text() == "old"


@pytest.mark.parametrize("enabled", [False, True])
def test_seam_adjustment_is_opt_in_and_moves_only_the_boundary(enabled):
    from character_dna.editors.converter.pipeline import ConversionPipeline
    from character_dna.utilities import get_head_to_body_edge_loop_mapping

    body = bpy.data.meshes.new("seam_body")
    base = np.full((30455, 3), 0.001, dtype=np.float32)
    body.from_pydata(base, [], [])
    obj = bpy.data.objects.new("seam_body", body)
    bpy.context.scene.collection.objects.link(obj)
    fake = SimpleNamespace(
        rows={"head": {"mapping": "head_lod0_mesh"}, "body": {"mapping": "body_lod0_mesh"}},
        targets={"head": np.zeros((24049, 3), dtype=np.float32), "body": base.copy()},
        objects={"body": obj},
        report={"warnings": []},
        operator=SimpleNamespace(_properties=SimpleNamespace(adjust_seam=enabled)),
    )
    try:
        transition = ConversionPipeline._seam_transition(fake)
        assert transition.changed is enabled
        transition.commit()
        expected = base.copy()
        if enabled:
            expected[list(get_head_to_body_edge_loop_mapping()["0"].values())] = 0
        np.testing.assert_array_equal(fake.targets["body"], expected)
        assert fake.report["seam"]["max_gap_m"] > 0
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(body)
