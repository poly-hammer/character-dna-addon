"""Component tabs and inactive solo filters must preserve scene visibility."""

import bpy
import pytest

from test_merge_split import character as character_fixture


character = character_fixture
pytestmark = pytest.mark.rig_profile(compact=True)


def presentation():
    return {
        obj.name: (obj.hide_get(), obj.hide_viewport, obj.hide_render)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    }


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
@pytest.mark.parametrize("mode", ["SEPARATE", "SCENE", "DNA"])
def test_component_tabs_and_mesh_filters_preserve_lod_visibility(character, mode):
    instance = character
    if mode != "SEPARATE":
        instance.converter_merge_mode = mode
        assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    assert not instance.raw_control_editor.solo_visibility
    assert not instance.shape_key_editor.solo_visibility
    assert not instance.shape_key_editor.shape_key_list
    for lod in (0, 3, 5):
        instance.view_options.active_lod = f"lod{lod}"
        # Preserve deliberate artist hiding too, rather than resetting the LOD.
        teeth = bpy.data.objects[f"{instance.name}_teeth_lod{lod}_mesh"]
        teeth.hide_set(True)
        expected = presentation()
        active = bpy.context.view_layer.objects.active
        for component in ("body", "head", "body", "head"):
            instance.editor_component = component
            assert instance.view_options.active_lod == f"lod{lod}"
            assert presentation() == expected, (mode, lod, component)
            assert bpy.context.view_layer.objects.active == active
        editor = instance.raw_control_editor
        editor.target_mesh_filter = str(instance.head_dna_reader.getMeshIndicesForLOD(0)[0])
        assert presentation() == expected
        editor.target_mesh_filter = "ALL"
        assert presentation() == expected


def test_raw_mesh_filter_still_updates_enabled_solo(character):
    instance = character
    editor = instance.raw_control_editor
    meshes = {
        instance.head_dna_reader.getMeshName(index): (index, obj)
        for index, obj in instance.head_mesh_index_lookup.items()
    }
    head_index, head = meshes["head_lod0_mesh"]
    teeth_index, teeth = meshes["teeth_lod0_mesh"]
    editor.target_mesh_filter = str(head_index)
    editor.solo_visibility = True
    assert not head.hide_get() and teeth.hide_get()
    editor.target_mesh_filter = str(teeth_index)
    assert head.hide_get() and not teeth.hide_get()
