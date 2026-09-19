"""End-to-end coverage for the Mesh-to-DNA Converter operator.

Mirrors the legacy ``test_convert_to_dna`` round trip, updated for the converter
editor: the operator now takes the head/body mesh pointers directly and fits the
chosen base DNA onto them. Running headless (``bpy.app.background``) the operator
takes its synchronous path, so the settle animation is committed immediately."""

import json
import shutil

from pathlib import Path

import bpy
import pytest

from character_dna.editors.converter.properties import get_converter_properties
from constants import TEST_DNA_FOLDER


@pytest.fixture
def base_dna_folder(temp_folder: Path):
    """Copy the Ada base head/body DNA into a temporary folder and register it as
    an Extra DNA Folder so the converter's ``base_dna`` enum can select it."""
    from character_dna.utilities import get_addon_preferences

    folder = temp_folder / "converter_e2e"
    folder.mkdir(parents=True, exist_ok=True)
    for component in ("head", "body"):
        shutil.copy2(TEST_DNA_FOLDER / "ada" / f"{component}.dna", folder / f"{component}.dna")

    preferences = get_addon_preferences()
    assert preferences is not None, "Addon preferences should be available"
    entry = preferences.extra_dna_folder_list.add()
    entry.folder_path = str(folder.absolute())
    try:
        yield folder
    finally:
        index = next(
            (i for i, e in enumerate(preferences.extra_dna_folder_list) if e.folder_path == str(folder.absolute())),
            None,
        )
        if index is not None:
            preferences.extra_dna_folder_list.remove(index)
        shutil.rmtree(folder, ignore_errors=True)


def test_convert_meshes_to_dna(
    load_mhc_conformed_topology_meshes, base_dna_folder, temp_folder: Path, monkeypatch: pytest.MonkeyPatch
):
    name = "TestMetaHuman01"
    output_folder = temp_folder / "converted_dna"
    output_folder.mkdir(parents=True, exist_ok=True)

    bpy.context.scene.unit_settings.scale_length = 1.0

    properties = get_converter_properties(bpy.context)
    properties.head_mesh = bpy.data.objects["head_lod0_mesh"]
    properties.body_mesh = bpy.data.objects["body_lod0_mesh"]
    properties.new_name = name
    properties.new_folder = str(output_folder)
    properties.base_dna = str(base_dna_folder.absolute())
    properties.validate_uvs = False
    properties.constrain_head_to_body = True

    # Interactive conversion yields to timers before finalization; background
    # conversion does not. Reproduce the deferred native binding at that boundary.
    from character_dna.editors.converter import utilities as converter_utilities
    from character_dna.runtime import controller, engine

    snap_components = converter_utilities.snap_components_to_wrap

    def snap_with_native_bindings(prepared):
        controller.reconcile()
        instance = bpy.context.scene.character_dna.rig_instance_list.get(name)
        assert engine.active(instance)
        if bpy.app.version < (5, 0, 0):
            assert instance.face_board.data.users > 1
        snap_components(prepared)

    monkeypatch.setattr(converter_utilities, "snap_components_to_wrap", snap_with_native_bindings)
    result = bpy.ops.character_dna.convert_to_dna()  # type: ignore[attr-defined]
    assert result == {"FINISHED"}, "Conversion operator should finish successfully"

    instance = bpy.context.scene.character_dna.rig_instance_list.get(name)  # type: ignore[attr-defined]
    assert instance is not None, "Rig instance should be created"
    assert instance.name == name, f"Instance name should be {name}"

    assert instance.head_rig is not None, "Head rig should be created"
    assert instance.head_mesh is not None, "Head mesh should be created"
    assert instance.head_dna_file_path, "Head DNA file path should be set"

    assert instance.body_rig is not None, "Body rig should be created"
    assert instance.body_mesh is not None, "Body mesh should be created"
    assert instance.body_dna_file_path, "Body DNA file path should be set"

    # The converter should write an ExportManifest.json alongside the DNA, like
    # the MetaHuman Creator DCC export.
    manifest_file = output_folder / "ExportManifest.json"
    assert manifest_file.exists(), "ExportManifest.json should be written alongside the DNA"
    manifest = json.loads(manifest_file.read_text())
    assert manifest["metaHumanName"] == name, "Manifest should record the converted MetaHuman name"

    # ``zero_shape_deltas`` defaults on, so the converted head DNA should carry no
    # blend shape deltas even though the base DNA does.
    from character_dna.dna_io import get_dna_reader

    def _blend_shape_delta_count(dna_path: Path) -> int:
        reader = get_dna_reader(dna_path)
        assert reader is not None, f"Should be able to read {dna_path}"
        return sum(
            len(reader.getBlendShapeTargetVertexIndices(mesh_index, target_index))
            for mesh_index in range(reader.getMeshCount())
            for target_index in range(reader.getBlendShapeTargetCount(mesh_index))
        )

    assert _blend_shape_delta_count(base_dna_folder / "head.dna") > 0, (
        "The base head DNA should ship blend shape deltas for this test to be meaningful"
    )
    assert _blend_shape_delta_count(output_folder / "head.dna") == 0, (
        "zero_shape_deltas should clear every blend shape delta in the converted head DNA"
    )
