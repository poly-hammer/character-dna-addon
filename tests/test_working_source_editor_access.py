"""Editor helpers must evaluate detached canonical meshes without exposing them."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.dna_io.misc import create_shape_key
from character_dna.editors.converter import scene_merge
from character_dna.editors.raw_control_editor.utilities import (
    _read_evaluated_mesh_vertices_local,
    create_target_mesh_for_dna_mesh,
    read_blender_mesh_in_dna_space,
)
from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor
from character_dna.editors.shape_key_editor.utilities import _preserve_mesh_mode
from character_dna.runtime import controller, engine
from character_dna.utilities.misc import deselect_all, set_hidden
from test_merge_split import character as character_fixture, coordinates


character = character_fixture


@pytest.fixture
def working_source():
    bpy.ops.wm.read_homefile(use_empty=True)
    mesh = bpy.data.meshes.new("WorkingSource")
    mesh.from_pydata([(0, 0, 0), (0, 1, 0), (0, 0, 1)], [], [(0, 1, 2)])
    source = bpy.data.objects.new("WorkingSource", mesh)
    source["dna_working_source"] = True
    source.shape_key_add(name="Basis")
    key = source.shape_key_add(name="head_lod0_mesh__test")
    key.data[0].co.x = 1.0
    key.value = 0.6
    try:
        yield source
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


def test_detached_sources_are_excluded_from_selection_and_visibility(working_source, caplog):
    visible = bpy.data.objects.new("Visible", None)
    bpy.context.scene.collection.objects.link(visible)
    visible.select_set(True)
    assert not set_hidden(working_source, False)
    deselect_all()
    assert not visible.select_get()
    assert working_source.name not in bpy.context.scene.objects
    assert "Skipping visibility" not in caplog.text


def test_raw_target_uses_evaluated_detached_geometry(working_source):
    instance = SimpleNamespace(
        name="DetachedRig",
        edit_target=SimpleNamespace(
            mesh_index_lookup={0: working_source},
            dna_reader=SimpleNamespace(getMeshName=lambda _index: "head_lod0_mesh"),
        ),
        get=lambda _name, default=None: default,
    )
    target = create_target_mesh_for_dna_mesh(instance, 0, "test")
    assert target.data.vertices[0].co.x == pytest.approx(0.6)
    assert target.name in bpy.context.scene.objects
    assert working_source.name not in bpy.context.scene.objects


@pytest.mark.parametrize("dna_space", [False, True], ids=["transfer", "match"])
def test_raw_operations_sample_detached_deformation(working_source, dna_space):
    if dna_space:
        positions = read_blender_mesh_in_dna_space(working_source, 100.0, 0.0)
        expected = 60.0
    else:
        positions = _read_evaluated_mesh_vertices_local(working_source)
        expected = 0.6
    assert positions[0][0] == pytest.approx(expected)
    assert working_source.name not in bpy.context.scene.objects


def test_shape_mode_scope_unlinks_sources_after_failure(working_source):
    active = bpy.data.objects.new("Active", None)
    bpy.context.scene.collection.objects.link(active)
    bpy.context.view_layer.objects.active = active
    with (
        pytest.raises(RuntimeError, match="Simulated transfer failure"),
        _preserve_mesh_mode(working_source) as mode,
    ):
        assert mode == "OBJECT"
        assert bpy.context.view_layer.objects.active == working_source
        assert working_source.name in bpy.context.scene.objects
        raise RuntimeError("Simulated transfer failure")
    assert working_source.name not in bpy.context.scene.objects
    assert bpy.context.view_layer.objects.active == active


@pytest.mark.parametrize("commit", [False, True], ids=["revert", "commit"])
def test_dna_shape_edit_keeps_sources_out_of_scene(character, commit):
    instance = character
    reader = instance.head_dna_reader
    mesh_index = next(index for index, obj in instance.head_mesh_index_lookup.items() if obj == instance.head_mesh)
    channel = reader.getBlendShapeChannelIndex(mesh_index, 0)
    with controller.authoring_operation(instance):
        key = create_shape_key(
            0,
            mesh_index,
            instance.head_mesh,
            reader,
            reader.getBlendShapeChannelName(channel),
            prefix=f"{reader.getMeshName(mesh_index)}__",
            linear_modifier=0.01,
        )
        name = key.name
        original = coordinates(key.data).copy()
        instance.destroy_references()
        instance.sync_shape_key_list()
    instance.shape_key_editor.shape_key_list_active_index = instance.shape_key_editor.shape_key_list.find(name)
    instance.converter_merge_mode = "DNA"
    old_names = [instance.head_mesh.name, instance.body_mesh.name]
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    assert all(bpy.data.objects.get(name) is None for name in old_names)
    editor = ShapeKeyEditor.for_instance(instance)
    editor.enter(bpy.context)
    proxy = scene_merge.proxy_for(instance)
    assert bpy.context.object == proxy
    bpy.ops.object.mode_set(mode="OBJECT")
    index = [item.value for item in proxy.data.attributes["DNA_head_vertex"].data].index(0)
    proxy.active_shape_key.data[index].co.x += 0.01
    if commit:
        editor.commit(bpy.context)
    else:
        editor.revert(bpy.context)
    assert scene_merge.proxy_for(instance) == proxy
    assert all(obj.name not in bpy.context.scene.objects for obj in (instance.head_mesh, instance.body_mesh))
    expected = original.copy()
    if commit:
        expected[0, 0] += 0.01
    np.testing.assert_allclose(
        coordinates(instance.head_mesh.data.shape_keys.key_blocks[name].data), expected, atol=2e-6
    )
    assert engine.binding_issues(instance) == []
