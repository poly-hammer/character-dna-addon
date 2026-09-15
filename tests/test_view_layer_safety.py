import bpy
import pytest

from character_dna.utilities import (
    is_collection_in_scene,
    move_to_collection,
    select_only,
    set_active,
    set_hidden,
    set_selected,
    switch_to_pose_mode,
)


@pytest.fixture
def empty_scene():
    """Provide a clean, empty scene with the addon enabled and reset afterward."""
    bpy.ops.wm.read_homefile(use_empty=True)
    yield bpy.context.scene
    bpy.ops.wm.read_homefile(use_empty=True)


def _armature_object(name: str) -> bpy.types.Object:
    return bpy.data.objects.new(name, bpy.data.armatures.new(name))


def _in_excluded_collection(scene: bpy.types.Scene, name: str) -> bpy.types.Object:
    """An object Blender refuses to hide, select or activate: linked to the scene but excluded."""
    collection = bpy.data.collections.new(f"{name}_collection")
    scene.collection.children.link(collection)
    scene_object = _armature_object(name)
    collection.objects.link(scene_object)
    bpy.context.view_layer.layer_collection.children[collection.name].exclude = True
    return scene_object


def _outside_the_scene(scene: bpy.types.Scene, name: str) -> bpy.types.Object:
    """An object in a collection that was never linked to the scene."""
    collection = bpy.data.collections.new(f"{name}_collection")
    scene_object = _armature_object(name)
    collection.objects.link(scene_object)
    return scene_object


@pytest.mark.parametrize("make_object", [_in_excluded_collection, _outside_the_scene])
def test_visibility_helpers_skip_objects_outside_the_view_layer(empty_scene, make_object):
    scene_object = make_object(empty_scene, "unreachable_rig")

    assert set_hidden(scene_object, True) is False
    assert set_selected(scene_object, True) is False
    assert set_active(scene_object) is False


def test_visibility_helpers_apply_to_objects_in_the_view_layer(empty_scene):
    scene_object = _armature_object("visible_rig")
    empty_scene.collection.objects.link(scene_object)

    assert set_selected(scene_object, True) is True
    assert scene_object.select_get() is True
    assert set_active(scene_object) is True
    assert bpy.context.view_layer.objects.active == scene_object
    assert set_hidden(scene_object, True) is True
    assert scene_object.hide_get() is True


def test_visibility_helpers_tolerate_no_object():
    assert set_hidden(None, True) is False
    assert set_selected(None, True) is False
    assert set_active(None) is False


def test_select_only_tolerates_an_object_outside_the_view_layer(empty_scene):
    unreachable = _in_excluded_collection(empty_scene, "excluded_rig")
    reachable = _armature_object("visible_rig")
    empty_scene.collection.objects.link(reachable)

    assert select_only(unreachable, reachable) is True
    assert reachable.select_get() is True
    assert bpy.context.view_layer.objects.active == reachable


def test_select_only_reports_when_nothing_could_be_activated(empty_scene):
    assert select_only(_in_excluded_collection(empty_scene, "excluded_rig")) is False


def test_switch_to_pose_mode_does_not_raise_outside_the_view_layer(empty_scene):
    switch_to_pose_mode(_in_excluded_collection(empty_scene, "excluded_rig"))

    assert bpy.context.mode == "OBJECT"


def test_move_to_collection_relinks_a_collection_left_outside_the_scene(empty_scene):
    orphan = bpy.data.collections.new("Ada")
    scene_object = _armature_object("Ada_head_rig")
    empty_scene.collection.objects.link(scene_object)
    assert not is_collection_in_scene(orphan)

    move_to_collection(scene_objects=[scene_object], collection_name="Ada", exclusively=True)

    assert is_collection_in_scene(orphan)
    assert orphan in scene_object.users_collection
    assert set_hidden(scene_object, True) is True
