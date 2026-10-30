"""The reusable artifact must not share scene state or writable paths between tests."""

import hashlib
import tempfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bpy
import numpy as np
import pytest

from character_dna.runtime import engine
from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigFactory, RigProfile, release_partitions, reset_scene


def test_narrow_profile_constructs_final_bindings_once(compact_dna, tmp_path):
    profile = RigProfile(asset=str(compact_dna), pose_bones=STATE_BONES, head_targets=(0,))
    installed = []
    install = engine.install

    def record(instance):
        installed.append(instance)
        return install(instance)

    try:
        with patch.object(engine, "install", record):
            instance = RigFactory._import(profile, tmp_path)
            assert installed == [instance]
        assert engine.active(instance) and not engine.binding_issues(instance)
        assert len(instance.head_rig.pose.bones) < instance.head_dna_reader.getJointCount() // 2
        assert len(instance.head_mesh.data.shape_keys.key_blocks) == 2
        # The final bindings must include the key created while installation
        # was deferred, not just the skeleton that existed at import time.
        assert sum(len(plan[1]) for plan in instance.component("head").shape_key_apply_plan) == 1
        assert engine.preview_shape(instance, 0, 0.6)
        bpy.context.view_layer.update()
        keys = instance.head_mesh.data.shape_keys.evaluated_get(bpy.context.view_layer.depsgraph)
        assert keys.key_blocks[1].value == pytest.approx(0.6)
    finally:
        reset_scene()


@pytest.mark.parametrize(
    "query", ["shape_key_blocks", "shape_key_blocks_by_name", "shape_key_apply_plan", "sync_shape_key_list"]
)
def test_bone_only_shape_queries_resolve_scene_once(empty_character_scene, query):
    from character_dna.dna_io import get_dna_reader, release_dna_handle
    from character_dna.rig_component import RigComponent
    from character_dna.rig_instance import RigInstance
    from constants import TEST_DNA_FOLDER

    reader = get_dna_reader(TEST_DNA_FOLDER / "ada/head.dna")
    instance = SimpleNamespace(
        name="ShapeLookup",
        head_dna_reader=reader,
        head_rig=None,
        data={},
        cache_key=lambda kind, descriptor: (kind, descriptor),
    )
    component = RigComponent(instance, "head")
    instance.edit_target = component
    instance.shape_key_editor = bpy.context.scene.character_dna.rig_instance_list.add().shape_key_editor
    try:
        # A head defines thousands of targets even when only its bones are
        # imported. Querying that empty scene must perform one lookup per batch.
        original = RigComponent.meshes_for_reader
        with patch.object(RigComponent, "meshes_for_reader", autospec=True, side_effect=original) as lookup:
            if query == "sync_shape_key_list":
                RigInstance.sync_shape_key_list(instance)
                assert not instance.shape_key_editor.shape_key_list
            else:
                assert not getattr(component, query)
            assert lookup.call_count == 1
        # The optimization is local to the batch. A later mesh remains visible
        # through the public lookup without persisting an empty scene cache.
        mesh = bpy.data.meshes.new("LateHead")
        mesh.from_pydata([(0, 0, 0)], [], [])
        obj = bpy.data.objects.new(f"ShapeLookup_{reader.getMeshName(0)}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.shape_key_add(name="Basis")
        name = f"{reader.getMeshName(0)}__{reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, 0))}"
        key = obj.shape_key_add(name=name)
        assert component.get_shape_key_block(0, name) == key
        if query == "sync_shape_key_list":
            RigInstance.sync_shape_key_list(instance)
            assert [item.name for item in instance.shape_key_editor.shape_key_list] == [name]
    finally:
        release_dna_handle(reader)


def test_partition_cleanup_preserves_preexisting_working_directories(monkeypatch):
    from character_dna.dna_io import document

    with (
        tempfile.TemporaryDirectory(prefix="character-dna-") as previous,
        tempfile.TemporaryDirectory(prefix="character-dna-") as current,
    ):
        paths = []
        for folder in (previous, current):
            path = Path(folder) / "document/rig/head.dna"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"working partition")
            paths.append(path.resolve())
        monkeypatch.setattr(document, "_leases", {path: (path, "head", "revision") for path in paths})
        release_partitions({paths[0]})
        assert paths[0].read_bytes() == b"working partition"
        assert set(document._leases) == {paths[0]}
        assert not Path(current).exists()


@pytest.mark.parametrize("mode", ["scene", "combined"])
@pytest.mark.parametrize("head_targets", [(), (0,)])
def test_merged_leases_own_document_and_partition_paths(rig_factory, tmp_path, mode, head_targets, compact_dna):
    from character_dna import shape_names
    from character_dna.editors.converter import scene_merge

    profile = RigProfile(asset=str(compact_dna), pose_bones=STATE_BONES, document_mode=mode, head_targets=head_targets)
    partitions = []
    original = None
    previous = None
    original_vertex = None
    for index in range(2):
        with rig_factory.lease(profile, tmp_path / str(index)) as instance:
            assert instance.components_merged
            assert scene_merge.proxy_for(instance) is not None
            scene_merge.prepare_export(instance, apply=False)
            current = {Path(instance.component(kind).dna_file_path) for kind in ("head", "body")}
            if partitions:
                assert not partitions[0].intersection(current)
            partitions.append(current)
            authoritative = Path(instance["combined_dna_path"] if mode == "combined" else instance.body_dna_file_path)
            content = authoritative.read_bytes()
            if original is None:
                original = content
            else:
                assert content == original
                assert previous.read_bytes() == b"changed previous lease"
            np.testing.assert_allclose(instance.body_rig.pose.bones["calf_l"].matrix_basis, np.eye(4), atol=1e-6)
            instance.body_rig.pose.bones["calf_l"].rotation_mode = "XYZ"
            instance.body_rig.pose.bones["calf_l"].rotation_euler.x = 0.7
            if head_targets:
                reader = instance.head_dna_reader
                channel = reader.getBlendShapeChannelIndex(0, 0)
                name = "head_lod0_mesh__" + reader.getBlendShapeChannelName(channel)
                keys = instance.head_mesh.data.shape_keys
                key = shape_names.find_key(keys, name)
                assert key is not None
                if original_vertex is None:
                    original_vertex = tuple(key.data[0].co)
                else:
                    assert tuple(key.data[0].co) == pytest.approx(original_vertex)
                assert engine.preview_shape(instance, channel, 0.65)
                bpy.context.view_layer.update()
                evaluated = keys.evaluated_get(bpy.context.evaluated_depsgraph_get())
                assert shape_names.find_key(evaluated, name).value == pytest.approx(0.65, abs=1e-5)
                key.data[0].co.x += 0.01
            assert engine.active(instance) and not engine.binding_issues(instance)
        if mode == "combined":
            from character_dna.dna_io import document

            assert all(not path.exists() and path not in document._leases for path in current)
            # A following lease must also work after the prior edit or another
            # document evicts its parsed component views.
            document._read_cache.clear()
        if index == 0:
            authoritative.write_bytes(b"changed previous lease")
            previous = authoritative


def test_rig_leases_restore_pristine_state_and_own_paths(rig_factory, tmp_path):
    profile = RigProfile(components=("body",))
    with rig_factory.lease(profile, tmp_path / "first") as instance:
        source = Path(instance.body_dna_file_path)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        bone = instance.body_rig.pose.bones["calf_l"]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler.x = 0.7
        instance.body_mesh.shape_key_add(name="OnlyFirstLease")
        bpy.context.scene.frame_set(17)
        bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
        visibility = [(c.name, c.is_visible, c.is_solo) for c in instance.body_rig.data.collections_all]
        instance.body_rig.data.collections[0].is_visible = not visibility[0][1]
        instance.output.folder_path = str(tmp_path / "custom-output")
    # Once released, even a corrupt previous input must not poison the artifact.
    source.write_bytes(b"first lease's intentionally changed DNA")
    with rig_factory.lease(profile, tmp_path / "second") as instance:
        second = Path(instance.body_dna_file_path)
        assert source != second and second.parent == tmp_path / "second"
        assert hashlib.sha256(second.read_bytes()).hexdigest() == before
        assert bpy.context.scene.frame_current == 1
        assert not bpy.context.scene.tool_settings.use_keyframe_insert_auto
        assert instance.body_mesh.data.shape_keys is None
        assert [(c.name, c.is_visible, c.is_solo) for c in instance.body_rig.data.collections_all] == visibility
        np.testing.assert_allclose(instance.body_rig.pose.bones["calf_l"].matrix_basis, np.eye(4), atol=1e-6)
        assert Path(instance.output.folder_path) == tmp_path / "second/output"
        assert all(Path(carrier["body_dna_file_path"]) == second for carrier in engine.carriers(instance))
        assert engine.active(instance) and not engine.binding_issues(instance)
    assert source.read_bytes() == b"first lease's intentionally changed DNA"


def test_failed_edit_does_not_leave_suspended_runtime_or_editor_cache(rig_factory, tmp_path):
    from character_dna.editors.swing_twist_editor.editor import SwingTwistEditor
    from character_dna.rig_instance import RigInstance
    from character_dna.runtime import controller

    profile = RigProfile(components=("body",))
    for index in range(3):
        with (
            pytest.raises(RuntimeError, match="test failure during edit"),
            rig_factory.lease(profile, tmp_path / str(index)) as instance,
        ):
            instance.editor_component = "body"
            SwingTwistEditor.for_instance(instance).enter(bpy.context)
            assert controller.is_suspended(instance)
            raise RuntimeError("test failure during edit")
        assert not RigInstance.data
        assert not engine._records and not engine._models
        assert not controller._suspended
        assert bpy.app.handlers.blend_import_post.count(controller._after_import) == 1
    with rig_factory.lease(profile, tmp_path / "recovered") as instance:
        assert not instance.swing_twist_editor.is_editing
        assert engine.active(instance)
