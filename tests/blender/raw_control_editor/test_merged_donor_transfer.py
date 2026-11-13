"""Copy the visible merged character into independent raw sculpt targets."""

import bmesh
import bpy
import numpy as np
import pytest

from character_dna.editors.converter.scene import merge
from character_dna.editors.raw_control_editor import utilities
from character_dna.editors.raw_control_editor.editor import RawControlEditor
from character_dna.utilities import get_active_rig_instance
from fixtures.rigs import RigProfile


@pytest.fixture
def character(tmp_path, request, rig_factory):
    marker = request.node.get_closest_marker("rig_profile")
    options = {"document_mode": "scene", **(marker.kwargs if marker else {})}
    if options.pop("compact", False):
        from fixtures.compact_dna import STATE_BONES

        options.update(asset=str(request.getfixturevalue("compact_dna")), pose_bones=STATE_BONES)
    with rig_factory.lease(RigProfile(**options), tmp_path) as instance:
        yield instance


def coordinates(data):
    values = np.empty(len(data) * 3, dtype=np.float32)
    data.foreach_get("co", values)
    return values.reshape(-1, 3)


def evaluated_coordinates(obj):
    bpy.context.view_layer.update()
    return coordinates(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)


def component_coordinates(proxy, source, kind):
    """Independent ID-based oracle, including both sides of the welded seam."""
    indices = np.array([item.value for item in proxy.data.attributes[f"DNA_{kind}_vertex"].data])
    valid = indices >= 0
    result = np.empty((len(source.data.vertices), 3), dtype=np.float32)
    result[indices[valid]] = evaluated_coordinates(proxy)[valid]
    transform = np.array(source.matrix_world.inverted() @ proxy.matrix_world)
    return result @ transform[:3, :3].T + transform[:3, 3]


def select_donor(proxy):
    bpy.context.view_layer.update()
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy
    assert proxy in bpy.context.selected_objects, (proxy.hide_get(), proxy.hide_viewport, proxy.hide_select)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_transfer_merged_donor_copies_visible_component_in_canonical_order(character, kind):
    instance = character
    instance.editor_component = kind
    props = instance.raw_control_editor
    props.raw_controls_active_index = props.raw_controls.find("default")
    editor = RawControlEditor.for_instance(instance)
    editor.enter(bpy.context)
    try:
        props.target_mesh_offset = True
        props.view_mode = "OBJECT"
        proxy = merge.proxy_for(instance)
        source = instance.edit_target.mesh
        target = props.target_meshes[0].scene_object
        other_targets = {
            row.scene_object: coordinates(row.scene_object.data.vertices) for row in props.target_meshes[1:]
        }
        bone = instance.edit_target.rig.pose.bones["FACIAL_L_LipCorner" if kind == "head" else "calf_l"]
        bone.location.z += 0.02
        instance.edit_target.rig.update_tag()
        key = proxy.shape_key_add(name="transfer_probe", from_mix=False)
        key.data[0].co.x += 0.015
        key.value = 1
        proxy.data.update()
        expected = component_coordinates(proxy, source, kind)
        initial = coordinates(target.data.vertices)
        assert np.max(np.linalg.norm(expected - initial, axis=1)) > 0.001
        source_before = coordinates(source.data.vertices)
        proxy_before = evaluated_coordinates(proxy)
        matrix = target.matrix_world.copy()
        select_donor(proxy)
        assert utilities.resolve_donor_mesh(instance, props) == proxy
        assert bpy.ops.character_dna.rce_transfer_shape_from_selected_mesh() == {"FINISHED"}
        np.testing.assert_allclose(coordinates(target.data.vertices), expected, atol=2e-6)
        assert target.matrix_world == matrix
        assert merge.proxy_for(instance) == proxy
        np.testing.assert_array_equal(coordinates(source.data.vertices), source_before)
        np.testing.assert_array_equal(evaluated_coordinates(proxy), proxy_before)
        for obj, values in other_targets.items():
            np.testing.assert_array_equal(coordinates(obj.data.vertices), values)
    finally:
        editor.revert(bpy.context)


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("mapping", ["reordered_ids", "component_uv", "ids_without_uvs"])
@pytest.mark.rig_profile(compact=True)
def test_merged_transfer_respects_mapping_transforms_selection_and_strength(character, kind, mapping):
    instance = character
    instance.editor_component = kind
    props = instance.raw_control_editor
    props.raw_controls_active_index = props.raw_controls.find("default")
    proxy = merge.proxy_for(instance)
    if mapping == "reordered_ids":
        select_donor(proxy)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.sort_elements(type="REVERSE", elements={"VERT"})
        bpy.ops.object.mode_set(mode="OBJECT")
    editor = RawControlEditor.for_instance(instance)
    editor.enter(bpy.context)
    try:
        props.target_mesh_offset = True
        props.view_mode = "OBJECT"
        props.mesh_operation_strength = 0.35
        source = instance.edit_target.mesh
        target = props.target_meshes[0].scene_object
        proxy.location = (0.3, -0.25, 0.15)
        proxy.rotation_euler.z = 0.18
        proxy.scale = (1.2, 0.8, 1.1)
        expected = component_coordinates(proxy, source, kind)
        if mapping == "component_uv":
            proxy.data.attributes.remove(proxy.data.attributes[f"DNA_{kind}_vertex"])
        elif mapping == "ids_without_uvs":
            for obj in (proxy, source, target):
                for layer in list(obj.data.uv_layers):
                    obj.data.uv_layers.remove(layer)
        initial = coordinates(target.data.vertices)
        matrix = target.matrix_world.copy()
        props.view_mode = "EDIT"
        bpy.ops.mesh.select_all(action="DESELECT")
        mesh = bmesh.from_edit_mesh(target.data)
        mesh.verts.ensure_lookup_table()
        selected = np.arange(len(mesh.verts)) % 3 == 0
        for vertex, select in zip(mesh.verts, selected, strict=True):
            vertex.select = bool(select)
        bmesh.update_edit_mesh(target.data, loop_triangles=False)
        name, count = utilities.transfer_shape_from_selected_mesh(props, proxy, instance)
        assert name == props.target_meshes[0].name
        assert count == selected.sum()
        assert bpy.context.mode == "EDIT_MESH"
        bpy.ops.object.mode_set(mode="OBJECT")
        blended = initial.copy()
        blended[selected] += (expected[selected] - initial[selected]) * 0.35
        np.testing.assert_allclose(coordinates(target.data.vertices), blended, atol=2e-6)
        assert target.matrix_world == matrix
    finally:
        editor.revert(bpy.context)


@pytest.mark.parametrize("failure", ["unrelated_target", "missing_mapping"])
@pytest.mark.rig_profile(compact=True)
def test_invalid_merged_transfer_leaves_target_unchanged(character, failure):
    props = character.raw_control_editor
    props.raw_controls_active_index = props.raw_controls.find("default")
    editor = RawControlEditor.for_instance(character)
    editor.enter(bpy.context)
    try:
        props.view_mode = "OBJECT"
        proxy = merge.proxy_for(character)
        if failure == "unrelated_target":
            props.target_meshes_active_index = props.target_meshes.find("teeth_lod0_mesh")
            message = "does not contain head mesh index"
        else:
            proxy.data.attributes.remove(proxy.data.attributes["DNA_head_vertex"])
            proxy.data.attributes.remove(proxy.data.attributes[merge.PARTITION])
            message = "component ownership.*missing"
        target = props.target_meshes[props.target_meshes_active_index].scene_object
        before = coordinates(target.data.vertices)
        with pytest.raises(utilities.RawControlEditorError, match=message):
            utilities.transfer_shape_from_selected_mesh(props, proxy, character)
        np.testing.assert_array_equal(coordinates(target.data.vertices), before)
    finally:
        editor.revert(bpy.context)


@pytest.mark.rig_profile(compact=True, document_mode="combined")
def test_merged_donor_transfer_recovers_saved_session_without_runtime_lookup(character, tmp_path, monkeypatch):
    props = character.raw_control_editor
    props.raw_controls_active_index = props.raw_controls.find("default")
    editor = RawControlEditor.for_instance(character)
    editor.enter(bpy.context)
    try:
        props.view_mode = "OBJECT"
        props.target_mesh_offset = True
        proxy = merge.proxy_for(character)
        proxy.name = "Renamed merged character"
        proxy["dna_merge_head"].name = "Renamed canonical head"
        key = proxy.shape_key_add(name="transfer_probe", from_mix=False)
        key.data[0].co.x += 0.02
        key.value = 1
        proxy.data.update()
        expected = component_coordinates(proxy, proxy["dna_merge_head"], "head")
        path = str(tmp_path / "raw-session.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
        bpy.ops.wm.open_mainfile(filepath=path)
        instance = get_active_rig_instance()
        editor = RawControlEditor.for_instance(instance)
        props = instance.raw_control_editor
        assert props.is_editing
        proxy = merge.proxy_for(instance)
        select_donor(proxy)
        # A fresh Blender process can reopen a suspended session before its DNA
        # readers have been loaded. Exercise that state even if this process
        # retained or restored a reader while reopening the test file.
        with monkeypatch.context() as patch:
            for name in ("dna_reader", "mesh_index_lookup"):
                patch.delitem(instance.data, instance.edit_target.cache_key(name), raising=False)
            assert not instance.edit_target.mesh_index_lookup
            assert bpy.ops.character_dna.rce_transfer_shape_from_selected_mesh() == {"FINISHED"}
        np.testing.assert_allclose(coordinates(props.target_meshes[0].scene_object.data.vertices), expected, atol=2e-6)
    finally:
        editor.revert(bpy.context)
