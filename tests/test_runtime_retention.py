"""Geometry edits retain native ownership; incompatible changes cannot reuse it."""

from pathlib import Path

import bpy
import pytest

from character_dna.dna_io import release_dna_handle
from character_dna.dna_io.misc import create_shape_key
from character_dna.editors.converter import scene_merge
from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor
from character_dna.runtime import controller, engine, retention
from test_merge_split import character as character_fixture


character = character_fixture


def prepare(instance, mode):
    with controller.authoring_operation(instance):
        reader = instance.head_dna_reader
        channel = reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, 0))
        key = create_shape_key(
            0, 0, instance.head_mesh, reader, channel, prefix="head_lod0_mesh__", linear_modifier=0.01
        )
        name = key.name
        instance.destroy_references()
        instance.sync_shape_key_list()
    if mode != "separate":
        instance.converter_merge_mode = "DNA" if mode == "combined" else "SCENE"
        assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    instance.shape_key_editor.shape_key_list_active_index = instance.shape_key_editor.shape_key_list.find(name)
    return ShapeKeyEditor.for_instance(instance)


def identities(instance):
    return {
        carrier["component"]: (
            carrier.as_pointer(),
            carrier["token"],
            id(engine._records[carrier.as_pointer()].get("model")),
        )
        for carrier in engine.carriers(instance)
    }


@pytest.mark.parametrize("mode", ["separate", "scene", "combined"])
def test_shape_commit_and_revert_preserve_bindings(character, mode):
    instance = character
    editor = prepare(instance, mode)
    before = identities(instance)
    for commit in (True, False, True):
        assert engine.preview_shape(instance, 0, 0.75)
        bpy.context.view_layer.update()
        editor.enter(bpy.context)
        assert retention.retained(instance)
        assert not engine.active(instance)
        assert identities(instance) == before
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = scene_merge.proxy_for(instance) if mode != "separate" else instance.head_mesh
        head_ids = obj.data.attributes.get("DNA_head_vertex")
        vertex = next(i for i, item in enumerate(head_ids.data) if item.value == 0) if head_ids else 0
        obj.active_shape_key.data[vertex].co.x += 0.001
        if commit:
            editor.commit(bpy.context)
        else:
            editor.revert(bpy.context)
        assert not retention.retained(instance)
        assert engine.active(instance)
        assert identities(instance) == before
        assert not engine.binding_issues(instance)
        assert all(not curve.mute for _owner, curve, _muted in retention._curves(engine.carriers(instance)))
        assert not any(carrier.get("authoring_paused") for carrier in engine.carriers(instance))
        assert not any(carrier.get("preview") for carrier in engine.carriers(instance))
    graph = bpy.context.evaluated_depsgraph_get()
    reused = {
        carrier["component"]: tuple(carrier.evaluated_get(graph)["outputs"]) for carrier in engine.carriers(instance)
    }
    controller.rebuild(instance)
    graph = bpy.context.evaluated_depsgraph_get()
    for carrier in engine.carriers(instance):
        assert tuple(carrier.evaluated_get(graph)["outputs"]) == pytest.approx(reused[carrier["component"]], abs=1e-5)


@pytest.mark.parametrize("change", ["behavior", "new_key", "undo"])
def test_incompatible_edits_rebuild(character, change):
    instance = character
    prepare(instance, "separate")
    before = identities(instance)
    was_active = controller.suspend(instance, preserve_bindings=True)
    assert was_active and retention.retained(instance)
    if change == "behavior":
        from character_dna.bindings import dna
        from character_dna.dna_io.document_edits import _reader
        from character_dna.dna_io.document import _atomic_write
        import os
        import tempfile

        path = Path(instance.head_dna_file_path)
        stamp = path.stat()
        instance.destroy_head()
        reader = _reader(path)
        # Use the SDK directly, bypassing addon invalidation as an external edit would.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "changed.dna"
            stream = dna.FileStream(str(output), dna.AccessMode_Write, dna.OpenMode_Binary, None)
            writer = dna.BinaryStreamWriter(stream)
            try:
                writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
                group = next(
                    index for index in range(reader.getJointGroupCount()) if len(reader.getJointGroupValues(index))
                )
                values = list(reader.getJointGroupValues(group))
                values[0] += 1.0
                writer.setJointGroupValues(group, values)
                writer.write()
                assert dna.Status.isOk()
            finally:
                for handle in (writer, stream, reader):
                    release_dna_handle(handle)
            _atomic_write(path, output.read_bytes())
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    elif change == "new_key":
        instance.head_mesh.shape_key_add(name="new_key")
    else:
        # Undo destroys all native/RNA session state; persisted paused drivers remain.
        controller.before_undo()
        controller.after_undo()
        assert not retention.retained(instance)
    controller.resume(instance, was_active)
    assert engine.active(instance)
    assert identities(instance) != before
    assert not engine.binding_issues(instance)
    assert not any(carrier.get("authoring_paused") for carrier in engine.carriers(instance))


def test_failed_document_commit_can_revert_without_losing_bindings(character, monkeypatch):
    from character_dna.dna_io import document_edits
    from character_dna.editors.shape_key_editor.core import ShapeKeyEditorError

    instance = character
    editor = prepare(instance, "combined")
    before = identities(instance)
    path = Path(instance["combined_dna_path"])
    original = path.read_bytes()
    editor.enter(bpy.context)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj = scene_merge.proxy_for(instance)
    vertex = next(i for i, item in enumerate(obj.data.attributes["DNA_head_vertex"].data) if item.value == 0)
    obj.active_shape_key.data[vertex].co.x += 0.001

    def fail(*args, **kwargs):
        raise OSError("Simulated native publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(document_edits, "write_targets", fail)
        # Publication occurs at the transaction boundary after the target writer.
        with pytest.raises((OSError, ShapeKeyEditorError), match="Simulated native publication failure"):
            editor.commit(bpy.context)
    assert path.read_bytes() == original
    assert instance.shape_key_editor.is_editing and retention.retained(instance)
    assert not engine.active(instance)
    editor.revert(bpy.context)
    assert identities(instance) == before
    assert engine.active(instance)


def test_retention_preserves_existing_driver_mutes(character):
    instance = character
    prepare(instance, "separate")
    owner, curve, _muted = retention._curves(engine.carriers(instance))[0]
    curve.mute = True
    owner.update_tag()
    before = identities(instance)
    active = controller.suspend(instance, preserve_bindings=True)
    assert retention.retained(instance)
    controller.resume(instance, active)
    assert curve.mute
    assert identities(instance) == before


@pytest.mark.parametrize("mode", ["scene", "combined"])
def test_shape_entry_rebuilds_when_bindings_cannot_be_retained(character, monkeypatch, mode):
    instance = character
    editor = prepare(instance, mode)
    before = identities(instance)
    # A file or native module change rejects reuse and drops stale authoring readers.
    monkeypatch.setattr(retention, "pause", lambda instance: False)
    editor.enter(bpy.context)
    assert editor.is_editing
    assert instance.head_dna_reader is not None
    assert not retention.retained(instance)
    editor.revert(bpy.context)
    assert engine.active(instance)
    assert identities(instance) != before
    assert not engine.binding_issues(instance)
    assert scene_merge.proxy_for(instance) is not None
