"""Editors and export share a component selection, including saved legacy files."""

from pathlib import Path
from unittest.mock import patch

import bpy
import pytest

from character_dna.dna_io import DNACalibrator, DNAExporter, get_dna_reader, release_dna_handle
from character_dna.rig_component import component_selection_locked
from fixtures.rigs import RigProfile


@pytest.fixture
def character(tmp_path, rig_factory):
    with rig_factory.lease(RigProfile(), tmp_path) as instance:
        (tmp_path / "output").mkdir()
        instance.output.auto_update_lods = False
        instance.output.align_head_and_body = False
        instance.output.run_validations = False
        yield instance


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("method", ["calibrate", "overwrite"])
def test_output_uses_shared_component_for_defaults_and_operator(character, kind, method):
    instance = character
    other = "body" if kind == "head" else "head"
    # A stale saved Output selection must not override the shared selection.
    instance.output["component"] = 1 if other == "body" else 0
    instance.editor_component = kind
    instance.output.method = method
    exporter_type = DNACalibrator if method == "calibrate" else DNAExporter
    for explicit, expected in ((None, kind), (other, other)):
        exporter = exporter_type(instance=instance, linear_modifier=0.01, component_type=explicit)
        try:
            assert exporter._component_type == expected
            assert exporter.source_dna_file == Path(getattr(instance, f"{expected}_dna_file_path"))
        finally:
            release_dna_handle(exporter._dna_writer)
            release_dna_handle(exporter._dna_reader)

    # This matrix owns routing, constructor defaults and the operator boundary.
    # Numerical exporter contracts live in the exporter/calibrator modules;
    # body operator publication is retained below as a real round trip.
    with patch.object(exporter_type, "run", autospec=True, return_value=(True, "Success", "Exported", None)) as run:
        assert bpy.ops.character_dna.export_selected_component() == {"FINISHED"}
    run.assert_called_once()
    invoked = run.call_args.args[0]
    assert type(invoked) is exporter_type
    assert invoked._component_type == kind
    assert invoked.source_dna_file == Path(getattr(instance, f"{kind}_dna_file_path"))
    assert invoked._target_dna_file == Path(instance.output.folder_path) / f"{kind}.dna"
    assert not invoked._include_textures
    release_dna_handle(invoked._dna_writer)
    release_dna_handle(invoked._dna_reader)


@pytest.mark.parametrize("method", ["calibrate", "overwrite"])
def test_body_operator_publication_preserves_joint_definition(character, method):
    instance = character
    instance.editor_component = "body"
    instance.output.method = method
    assert bpy.ops.character_dna.export_selected_component() == {"FINISHED"}
    folder = Path(instance.output.folder_path)
    assert (folder / "body.dna").exists()
    assert not (folder / "head.dna").exists()
    reader = get_dna_reader(folder / "body.dna")
    try:
        source = instance.body_dna_reader
        assert reader.getJointCount() == source.getJointCount()
        assert [reader.getJointName(index) for index in range(reader.getJointCount())] == [
            source.getJointName(index) for index in range(source.getJointCount())
        ]
    finally:
        release_dna_handle(reader)


def test_legacy_output_selection_and_session_locks(minimal_character):
    instance = minimal_character
    assert "component" not in instance.output.bl_rna.properties
    if "editor_component" in instance:
        del instance["editor_component"]
    instance.output["component"] = 1
    assert instance.editor_component == "body"
    instance.editor_component = "head"
    assert instance.editor_component == "head"
    for name in ("raw_control_editor", "rbf_editor", "shape_key_editor"):
        editor = getattr(instance, name)
        editor.is_editing = True
        try:
            assert component_selection_locked(instance)
            instance.editor_component = "body"
            assert instance.editor_component == "head"
        finally:
            editor.is_editing = False
    instance["native_editor_resume"] = True
    try:
        assert component_selection_locked(instance)
        instance.editor_component = "body"
        assert instance.editor_component == "head"
    finally:
        del instance["native_editor_resume"]
    instance.editor_component = "body"
    assert instance.editor_component == "body"
    assert not component_selection_locked(instance)


@pytest.mark.parametrize("pro_visible", [True, False])
def test_component_switch_refreshes_output_in_both_editions(minimal_character, monkeypatch, pro_visible):
    from character_dna import utilities

    instance = minimal_character
    monkeypatch.setattr(utilities, "pro_features_visible", lambda: pro_visible)
    instance.editor_component = "head"
    head_item = instance.output.head_item_list[0]
    head_item.include = False
    instance.output.body_item_list.clear()
    instance.editor_component = "body"
    assert instance.editor_component == "body"
    assert instance.body_mesh in {item.scene_object for item in instance.output.body_item_list}
    assert not head_item.include
