"""DNA merges remove scene sources; splits recreate canonical meshes from DNA."""

from pathlib import Path

import bpy
import numpy as np
import pytest

from test_merge_split import character as character_fixture, coordinates

from character_dna import shape_names
from character_dna.dna_io.importer import DNAImporter
from character_dna.dna_io.misc import create_shape_key
from character_dna.editors.converter import scene_merge, source_views
from character_dna.runtime import controller, engine


character = character_fixture


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
def test_dna_merge_removes_scene_objects_and_split_imports_new_meshes(character, monkeypatch):
    instance = character
    instance.converter_merge_mode = "DNA"
    originals = {obj.name: obj for _, sources, _ in scene_merge.surface_pairs(instance) for obj in sources.values()}
    expected = {name: coordinates(obj.data.vertices) for name, obj in originals.items()}
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    assert instance.view_options.active_lod == "lod0"
    for name, old in originals.items():
        assert name not in bpy.context.scene.objects and name not in bpy.data.objects
        with pytest.raises(ReferenceError):
            _ = old.name
    for proxy in scene_merge.surfaces(instance).values():
        for kind in ("head", "body"):
            working = proxy[f"dna_merge_{kind}"]
            assert source_views.is_working(working) and not working.users_scene
    assert not engine.binding_issues(instance)
    imported = []
    create = DNAImporter.create_mesh_object

    def record(self, lod_index, mesh_name):
        imported.append(mesh_name)
        return create(self, lod_index, mesh_name)

    monkeypatch.setattr(DNAImporter, "create_mesh_object", record)
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    assert len(imported) == len(originals)
    for name, positions in expected.items():
        obj = bpy.context.scene.objects[name]
        np.testing.assert_allclose(coordinates(obj.data.vertices), positions, atol=1e-6, rtol=0)
        assert not source_views.is_working(obj)
        assert obj.data.shape_keys is None
    assert not scene_merge.surfaces(instance)
    assert not engine.binding_issues(instance)
    assert all(Path(instance.component(kind).dna_file_path).exists() for kind in ("head", "body"))


def test_dna_split_preserves_object_settings_and_loaded_key_state(character):
    instance = character
    head = instance.head_mesh
    head["user_note"] = "Keep the sculpt setup"
    head.constraints.new("LIMIT_LOCATION").name = "User constraint"
    modifier = head.modifiers.new("User smooth", "SMOOTH")
    modifier.show_viewport = False
    head.vertex_groups.new(name="Sculpt mask").add([0], 0.25, "REPLACE")
    head.keyframe_insert("location", frame=1)
    object_action = head.animation_data.action
    reader = instance.head_dna_reader
    index = head["ca_dna_mesh_index"]
    channel = reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(index, 0))
    with controller.authoring_operation(instance):
        key = create_shape_key(0, index, head, reader, channel, prefix="head_lod0_mesh__", linear_modifier=0.01)
        key.interpolation = "KEY_CARDINAL"
        key.lock_shape = True
        custom = head.shape_key_add(name="head_lod0_mesh__scene_only", from_mix=False)
        custom.data[0].co.x += 0.01
        custom.keyframe_insert("value", frame=1)
        key_action = head.data.shape_keys.animation_data.action
        expected_custom = coordinates(custom.data)
    instance.converter_merge_mode = "DNA"
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    head = instance.head_mesh
    assert head["user_note"] == "Keep the sculpt setup"
    assert head.constraints["User constraint"].type == "LIMIT_LOCATION"
    assert not head.modifiers["User smooth"].show_viewport
    assert head.vertex_groups["Sculpt mask"].weight(0) == pytest.approx(0.25)
    assert head.animation_data.action == object_action
    key = shape_names.find_key(head.data.shape_keys, f"head_lod0_mesh__{channel}")
    assert key.interpolation == "KEY_CARDINAL" and key.lock_shape
    assert head.data.shape_keys.animation_data.action == key_action
    custom = shape_names.find_key(head.data.shape_keys, "head_lod0_mesh__scene_only")
    np.testing.assert_allclose(coordinates(custom.data), expected_custom, atol=1e-6, rtol=0)
