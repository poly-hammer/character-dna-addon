"""Editors and export share a component selection, including saved legacy files."""

import shutil

from pathlib import Path

import bpy
import pytest

from character_dna.dna_io import DNACalibrator, DNAExporter, get_dna_reader, release_dna_handle
from character_dna.rig_component import component_selection_locked
from character_dna.utilities import get_active_rig_instance
from constants import TEST_DNA_FOLDER


@pytest.fixture
def character(tmp_path):
    bpy.ops.wm.read_homefile(use_empty=True)
    for name in ("head.dna", "body.dna", "ExportManifest.json"):
        shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, tmp_path / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / "head.dna"),
        include_body=True,
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    )
    instance = get_active_rig_instance()
    output = tmp_path / "output"
    output.mkdir()
    instance.output.folder_path = str(output)
    instance.output.auto_update_lods = False
    instance.output.align_head_and_body = False
    instance.output.run_validations = False
    yield instance
    bpy.ops.wm.read_homefile(use_empty=True)


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

    assert bpy.ops.character_dna.export_selected_component() == {"FINISHED"}
    folder = Path(instance.output.folder_path)
    assert (folder / f"{kind}.dna").exists()
    assert not (folder / f"{other}.dna").exists()
    reader = get_dna_reader(folder / f"{kind}.dna")
    try:
        source = instance.component(kind).dna_reader
        assert reader.getJointCount() == source.getJointCount()
        assert reader.getJointName(reader.getJointCount() - 1) == source.getJointName(source.getJointCount() - 1)
    finally:
        release_dna_handle(reader)


def test_legacy_output_selection_and_session_locks(character):
    instance = character
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
def test_component_switch_refreshes_output_in_both_editions(character, monkeypatch, pro_visible):
    from character_dna import utilities

    instance = character
    monkeypatch.setattr(utilities, "pro_features_visible", lambda: pro_visible)
    instance.editor_component = "head"
    head_item = instance.output.head_item_list[0]
    head_item.include = False
    instance.output.body_item_list.clear()
    instance.editor_component = "body"
    assert instance.editor_component == "body"
    assert instance.body_mesh in {item.scene_object for item in instance.output.body_item_list}
    assert not head_item.include
