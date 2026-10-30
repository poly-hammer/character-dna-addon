"""DNA conversion history must recover overwritten files and component ownership."""

import json

from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from test_merge_split import character as character_fixture, coordinates

from character_dna.dna_io import document
from character_dna.editors.backup_manager import core as backups
from character_dna.editors.converter import merge_split, scene_merge
from character_dna.runtime import controller, engine
from character_dna.utilities import get_addon_preferences


character = character_fixture


@pytest.mark.parametrize("case", ["common", "different", "explicit", "combined"])
def test_output_directory_defaults(tmp_path, case):
    head = tmp_path / "head.dna"
    body = tmp_path / ("elsewhere/body.dna" if case in {"different", "explicit"} else "body.dna")
    combined = str(tmp_path / "published/Ada.dna") if case == "combined" else None
    instance = SimpleNamespace(
        output=SimpleNamespace(folder_path=str(tmp_path / "output") if case == "explicit" else ""),
        component=lambda kind: SimpleNamespace(dna_file_path=str(head if kind == "head" else body)),
        get=lambda key: combined if key == "combined_dna_path" else None,
    )
    if case == "different":
        with pytest.raises(ValueError, match="different folders"):
            merge_split._output_directory(instance)
    else:
        expected = tmp_path / {"common": "", "explicit": "output", "combined": "published"}[case]
        assert merge_split._output_directory(instance) == expected


def configure_backups(monkeypatch, tmp_path):
    monkeypatch.setattr(backups, "is_auto_backup_enabled", lambda: True)
    monkeypatch.setattr(backups, "get_max_backups", lambda: 50)
    monkeypatch.setattr(get_addon_preferences().backup_manager, "folder_path", str(tmp_path / "history"))


def entries(instance):
    return {
        entry.backup_type: (
            entry.backup_id,
            json.loads((Path(entry.folder_path) / "metadata.json").read_text()),
        )
        for entry in instance.backup_manager.dna_backup_list
    }


def test_same_folder_overwrite_and_restore_both_document_modes(character, tmp_path, monkeypatch):
    configure_backups(monkeypatch, tmp_path)
    instance = character
    instance.converter_merge_mode = "DNA"
    instance.output.folder_path = ""
    original_paths = {kind: Path(instance.component(kind).dna_file_path) for kind in document.PARTITIONS}
    original = {kind: path.read_bytes() for kind, path in original_paths.items()}
    history = backups.get_backup_folder(instance)
    combined = tmp_path / f"{instance.name}.dna"
    combined.write_bytes(b"previous destination")
    manual = backups.create_backup(instance, backups.BackupType.MANUAL)
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    published = combined.read_bytes()
    assert instance["combined_dna_path"] == str(combined)
    assert backups.get_backup_folder(instance) == history
    saved = entries(instance)
    pre_merge, pre = saved[backups.BackupType.PRE_DNA_MERGE.value]
    post_merge, post = saved[backups.BackupType.POST_DNA_MERGE.value]
    assert pre["operation_id"] == post["operation_id"]
    assert backups.restore_backup(instance, pre_merge)
    assert not instance.get("combined_dna_path") and not scene_merge.surfaces(instance)
    assert combined.read_bytes() == b"previous destination"
    assert all(Path(instance.component(kind).dna_file_path) == path for kind, path in original_paths.items())
    assert backups.restore_backup(instance, post_merge)
    assert combined.read_bytes() == published and scene_merge.surfaces(instance)
    # The former component files can have independent contents while the rig uses its document.
    for kind, path in original_paths.items():
        path.write_bytes(f"previous {kind} destination".encode())
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    split = {kind: path.read_bytes() for kind, path in original_paths.items()}
    assert not instance.get("combined_dna_path") and not scene_merge.surfaces(instance)
    assert all(value.startswith(b"DNA") for value in split.values())
    saved = entries(instance)
    pre_split, pre = saved[backups.BackupType.PRE_DNA_SPLIT.value]
    post_split, post = saved[backups.BackupType.POST_DNA_SPLIT.value]
    assert pre["operation_id"] == post["operation_id"]
    assert backups.restore_backup(instance, pre_split)
    assert instance.get("combined_dna_path") and scene_merge.surfaces(instance)
    assert all(path.read_bytes() == f"previous {kind} destination".encode() for kind, path in original_paths.items())
    assert backups.restore_backup(instance, post_split)
    assert not instance.get("combined_dna_path") and not scene_merge.surfaces(instance)
    assert all(path.read_bytes() == split[kind] for kind, path in original_paths.items())
    # Ordinary backups created before conversion remain reachable and restore the old mode too.
    assert backups.restore_backup(instance, post_merge)
    assert backups.restore_backup(instance, manual)
    assert not instance.get("combined_dna_path") and not scene_merge.surfaces(instance)
    assert all(path.read_bytes() == original[kind] for kind, path in original_paths.items())
    backups.sync_backup_list_with_disk(instance)
    assert len(instance.backup_manager.dna_backup_list) == 5
    assert backups.get_backup_folder(instance) == history
    assert not engine.binding_issues(instance)


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_backup_failure_leaves_original_files_and_mode(character, tmp_path, monkeypatch, phase):
    configure_backups(monkeypatch, tmp_path)
    instance = character
    instance.output.folder_path = ""
    instance.converter_merge_mode = "DNA"
    with controller.authoring_operation(instance):
        head = instance.head_mesh
        head.data.vertices[0].co.x += 0.012
        head.shape_key_add(name="Basis")
        key = head.shape_key_add(name="head_lod0_mesh__scene_only")
        key.data[1].co.z += 0.017
        original_vertices = coordinates(head.data.vertices)
        original_key = coordinates(key.data)
        instance.destroy_references()
        instance.sync_shape_key_list()
    destination = tmp_path / f"{instance.name}.dna"
    destination.write_bytes(b"original destination")
    create = backups.create_backup
    rejected = backups.BackupType.PRE_DNA_MERGE if phase == "pre" else backups.BackupType.POST_DNA_MERGE

    def fail(instance, backup_type, *args, **kwargs):
        return None if backup_type == rejected else create(instance, backup_type, *args, **kwargs)

    monkeypatch.setattr(backups, "create_backup", fail)
    messages = []
    operation = SimpleNamespace(report=lambda _level, message: messages.append(message))
    assert merge_split.MergeComponents.execute(operation, bpy.context) == {"CANCELLED"}
    assert any("Could not create" in message for message in messages)
    assert destination.read_bytes() == b"original destination"
    assert not instance.get("combined_dna_path") and not scene_merge.surfaces(instance)
    restored = instance.head_mesh
    assert restored.name in bpy.context.scene.objects
    np.testing.assert_array_equal(coordinates(restored.data.vertices), original_vertices)
    assert restored.data.shape_keys is not None
    assert "head_lod0_mesh__scene_only" in restored.data.shape_keys.key_blocks
    np.testing.assert_array_equal(
        coordinates(restored.data.shape_keys.key_blocks["head_lod0_mesh__scene_only"].data), original_key
    )
    assert not engine.binding_issues(instance)
    saved = entries(instance)
    assert backups.BackupType.POST_DNA_MERGE.value not in saved
    assert len(saved) == (phase == "post")


def test_retention_keeps_pairs_and_unique_ids(minimal_character, tmp_path, monkeypatch):
    from character_dna.editors.backup_manager.operators import track

    configure_backups(monkeypatch, tmp_path)
    instance = minimal_character
    # Retention consumes file records and paired operation IDs, not DNA syntax.
    # Complete merge/split and binding recovery remain covered by the real-rig
    # round trip and both failure phases immediately above this policy test.
    for kind in document.PARTITIONS:
        path = tmp_path / f"{kind}.dna"
        path.write_bytes(f"original {kind}".encode())
        instance[f"{kind}_dna_file_path"] = str(path)
    instance.converter_merge_mode = "DNA"
    monkeypatch.setattr(backups, "get_max_backups", lambda: 1)
    first = backups.create_backup(instance, backups.BackupType.MANUAL)
    second = backups.create_backup(instance, backups.BackupType.MANUAL)
    assert first != second
    with track(instance, "MERGE", []):
        pass
    assert len(instance.backup_manager.dna_backup_list) == 4
    merge_pair = entries(instance)
    assert (
        merge_pair[backups.BackupType.PRE_DNA_MERGE.value][1]["operation_id"]
        == merge_pair[backups.BackupType.POST_DNA_MERGE.value][1]["operation_id"]
    )
    with track(instance, "SPLIT", []):
        pass
    backups.sync_backup_list_with_disk(instance)
    saved = entries(instance)
    assert backups.BackupType.PRE_DNA_MERGE.value not in saved
    assert backups.BackupType.PRE_DNA_SPLIT.value in saved and backups.BackupType.POST_DNA_SPLIT.value in saved
    assert len(instance.backup_manager.dna_backup_list) == 4
    assert (
        saved[backups.BackupType.PRE_DNA_SPLIT.value][1]["operation_id"]
        == saved[backups.BackupType.POST_DNA_SPLIT.value][1]["operation_id"]
    )
    assert (
        saved[backups.BackupType.PRE_DNA_SPLIT.value][1]["operation_id"]
        != merge_pair[backups.BackupType.PRE_DNA_MERGE.value][1]["operation_id"]
    )


def test_file_restore_rolls_back_partial_overwrite(tmp_path, monkeypatch):
    first, second = tmp_path / "head.dna", tmp_path / "body.dna"
    created = tmp_path / "new.dna"
    first.write_bytes(b"old head")
    second.write_bytes(b"old body")
    write = document._atomic_write

    def fail(path, content):
        if path == second and content == b"new body":
            raise OSError("Simulated second file failure")
        write(path, content)

    monkeypatch.setattr(document, "_atomic_write", fail)
    with pytest.raises(OSError, match="second file"):
        document.restore_files({first: b"new head", created: b"new file", second: b"new body"})
    assert first.read_bytes() == b"old head" and second.read_bytes() == b"old body"
    assert not created.exists()


def test_backup_files_keep_same_basename_components_and_missing_destinations(tmp_path):
    paths = {kind: tmp_path / kind / "Ada.dna" for kind in document.PARTITIONS}
    for kind, path in paths.items():
        path.parent.mkdir()
        path.write_bytes(kind.encode())
    folder = tmp_path / "snapshot"
    folder.mkdir()
    missing = tmp_path / "combined.dna"
    instance = SimpleNamespace(
        head_dna_file_path=str(paths["head"]), body_dna_file_path=str(paths["body"]), get=lambda _key: None
    )
    records = backups._copy_backup_files(instance, folder, [*paths.values(), missing])
    assert len(records) == 3
    for kind, path in paths.items():
        assert (folder / f"{kind}.dna").read_bytes() == kind.encode()
        assert {"path": str(path), "file": f"{kind}.dna"} in records
    assert {"path": str(missing), "file": None} in records


@pytest.mark.parametrize("enabled", [False, True])
def test_failed_split_restores_overwritten_outputs(character, tmp_path, monkeypatch, enabled):
    configure_backups(monkeypatch, tmp_path)
    monkeypatch.setattr(backups, "is_auto_backup_enabled", lambda: enabled)
    instance = character
    instance.output.folder_path = ""
    instance.converter_merge_mode = "DNA"
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    combined = Path(instance["combined_dna_path"])
    original = combined.read_bytes()
    destinations = {kind: tmp_path / f"{kind}.dna" for kind in document.PARTITIONS}
    for kind, path in destinations.items():
        path.write_bytes(f"old {kind} destination".encode())
    create = backups.create_backup
    if enabled:
        monkeypatch.setattr(
            backups,
            "create_backup",
            lambda instance, backup_type, **kwargs: None
            if backup_type == backups.BackupType.POST_DNA_SPLIT
            else create(instance, backup_type, **kwargs),
        )
    else:
        write = document._atomic_write

        def fail(path, content):
            if path == destinations["body"] and content.startswith(b"DNA"):
                raise OSError("Simulated second split write failure")
            write(path, content)

        monkeypatch.setattr(document, "_atomic_write", fail)
    messages = []
    operation = SimpleNamespace(report=lambda _level, message: messages.append(message))
    assert merge_split.SplitComponents.execute(operation, bpy.context) == {"CANCELLED"}
    assert any(("Could not create" if enabled else "second split write") in message for message in messages)
    assert instance["combined_dna_path"] == str(combined)
    assert combined.read_bytes() == original and scene_merge.surfaces(instance)
    assert all(path.read_bytes() == f"old {kind} destination".encode() for kind, path in destinations.items())
    assert not engine.binding_issues(instance)
    saved = entries(instance)
    assert backups.BackupType.POST_DNA_SPLIT.value not in saved
    assert len(saved) == (3 if enabled else 0)
