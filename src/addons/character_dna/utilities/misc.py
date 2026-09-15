# standard library imports
import contextlib
import hashlib
import importlib
import json
import logging
import math
import re
import subprocess
import sys
import tomllib
import uuid

from collections.abc import Callable, Generator
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import addon_utils

# third party imports
import bpy

from mathutils import Vector

# local imports
from ..constants import (
    ADDON_IDS,
    DEFAULT_UV_TOLERANCE,
    EXCLUDED_FACE_BOARD_CONTROLS,
    EYE_AIM_BONES,
    FACE_BOARD_FILE_PATH,
    FACE_BOARD_NAME,
    FACE_BOARD_SWITCHES,
    FACE_GUI_EMPTIES,
    HEAD_TEXTURE_LOGIC_NODE_LABEL,
    INVALID_NAME_CHARACTERS_REGEX,
    LEGACY_DATA_KEYS,
    MATERIALS_FILE_PATH,
    MIGRATABLE_DATA_KEYS,
    NUMBER_OF_HEAD_LODS,
    SCRIPTS_FOLDER,
    TEMP_FOLDER,
    ToolInfo,
)
from ..rig_instance import begin_render, end_render
from ..typing import *  # noqa: F403
from . import get_active_rig_instance


logger = logging.getLogger(__name__)

# Distance below which two object origins are treated as already coincident.
_ORIGIN_TOLERANCE = 1e-6


def exclude_rig_instance_evaluation(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        window_manager_properties = get_addon_window_manager_properties()
        window_manager_properties.evaluate_dependency_graph = False
        result = func(*args, **kwargs)
        window_manager_properties.evaluate_dependency_graph = True
        return result

    return wrapper


def get_current_context() -> dict[str, Any]:
    object_contexts = {}
    if not bpy.context.scene:
        return {}
    for scene_object in bpy.context.scene.objects:
        active_action_name = ""
        if scene_object.animation_data and scene_object.animation_data.action:
            active_action_name = scene_object.animation_data.action.name

        object_contexts[scene_object.name] = {
            "hide": scene_object.hide_get(),
            "hide_viewport": scene_object.hide_viewport,
            "select": scene_object.select_get(),
            "active_action": active_action_name,
            "show_instancer_for_render": scene_object.show_instancer_for_render,
        }

    active_object = None
    if bpy.context.active_object:
        active_object = bpy.context.active_object.name

    return {
        "mode": getattr(bpy.context, "mode", "OBJECT"),
        "objects": object_contexts,
        "active_object": active_object,
        "current_frame": bpy.context.scene.frame_current,
        "cursor_location": bpy.context.scene.cursor.location,
    }


def set_context(context: dict[str, Any]) -> None:
    mode = context.get("mode", "OBJECT")
    active_object_name = context.get("active_object")
    object_contexts = context.get("objects", {})
    for object_name, attributes in object_contexts.items():
        scene_object = bpy.data.objects.get(object_name)
        if scene_object:
            set_hidden(scene_object, attributes.get("hide", False))
            scene_object.hide_viewport = attributes.get("hide_viewport", False)
            set_selected(scene_object, attributes.get("select", False))

            active_action = attributes.get("active_action")
            if active_action and scene_object.animation_data:
                scene_object.animation_data.action = bpy.data.actions.get(active_action)

            scene_object.show_instancer_for_render = attributes.get("show_instancer_for_render", False)

    # set the active object
    if active_object_name:
        set_active(bpy.data.objects.get(active_object_name))

    # set the mode
    if bpy.context.mode != mode:
        # Note:
        # When the mode context is read in edit mode it can be 'EDIT_ARMATURE' or 'EDIT_MESH', even though you
        # are only able to set the context to 'EDIT' mode. Thus, if 'EDIT' was read from the mode context, the mode
        # is set to edit.
        if "EDIT" in mode:
            mode = "EDIT"
        bpy.ops.object.mode_set(mode=mode)

    if bpy.context.scene:
        # set the current frame
        bpy.context.scene.frame_set(context.get("current_frame", 0))

        # set the cursor location
        bpy.context.scene.cursor.location = context.get("cursor_location", Vector((0, 0, 0)))


def preserve_context(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        window_manager_properties = get_addon_window_manager_properties()
        window_manager_properties.evaluate_dependency_graph = False
        context = get_current_context()
        result = func(*args, **kwargs)
        window_manager_properties.evaluate_dependency_graph = True
        set_context(context)
        return result

    return wrapper


@contextlib.contextmanager
def preserved_context() -> Generator[dict[str, Any], None, None]:
    """Context manager that snapshots the current scene context on entry and
    restores it on exit. Disables dependency graph evaluation for the duration
    so that intermediate mode/selection changes do not trigger rig evaluation.

    Yields the snapshotted context dict so it can be inspected or manipulated
    before it is restored on exit.

    Usage::

        with preserved_context() as context:
            switch_to_edit_mode(some_object)
            # ... inspect or mutate ``context`` as needed ...
        # scene context (mode, selection, active object, frame, cursor) is
        # automatically restored here.
    """
    window_manager_properties = get_addon_window_manager_properties()
    window_manager_properties.evaluate_dependency_graph = False
    context = get_current_context()
    try:
        yield context
    finally:
        set_context(context)
        window_manager_properties.evaluate_dependency_graph = True


def set_hidden(scene_object: bpy.types.Object | None, state: bool) -> bool:
    """Hide or unhide the object, skipping it when it has no base in the view layer.

    Blender raises on an object outside the view layer rather than treating it as a no-op, and a
    character can end up there via an excluded collection or a collection left out of the scene.
    The attempt is what reports it: ``view_layer.objects`` is not synced until the depsgraph runs,
    so a membership pre-check would skip objects Blender would have accepted.
    """
    if not scene_object:
        return False
    try:
        scene_object.hide_set(state)
    except RuntimeError as error:
        logger.warning(f'Skipping visibility change on "{scene_object.name}": {error}')
        return False
    return True


def set_selected(scene_object: bpy.types.Object | None, state: bool) -> bool:
    """Select or deselect the object, skipping it when it has no base in the view layer."""
    if not scene_object:
        return False
    try:
        scene_object.select_set(state)
    except RuntimeError as error:
        logger.warning(f'Skipping selection of "{scene_object.name}": {error}')
        return False
    return True


def set_active(scene_object: bpy.types.Object | None) -> bool:
    """Make the object active, skipping it when it has no base in the view layer."""
    if not scene_object or not bpy.context.view_layer:
        return False
    try:
        bpy.context.view_layer.objects.active = scene_object
    except RuntimeError as error:
        logger.warning(f'Skipping active object change to "{scene_object.name}": {error}')
        return False
    return True


def deselect_all():
    for scene_object in bpy.data.objects:
        scene_object.select_set(False)


def select_only(*scene_object: bpy.types.Object) -> bool:
    """Select the given objects and make the last reachable one active, reporting whether any was."""
    deselect_all()
    activated = False
    for _scene_object in scene_object:
        set_selected(_scene_object, True)
        activated = set_active(_scene_object) or activated
    return activated


def switch_to_object_mode():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def switch_to_edit_mode(*scene_object: bpy.types.Object):
    select_only(*scene_object)
    switch_to_object_mode()
    bpy.ops.object.mode_set(mode="EDIT")


def switch_to_sculpt_mode(*scene_object: bpy.types.Object):
    select_only(*scene_object)
    switch_to_object_mode()
    bpy.ops.object.mode_set(mode="SCULPT")


def switch_to_bone_edit_mode(*armature_object: bpy.types.Object):
    # Switch to edit mode so we can get edit bone data
    if bpy.context.mode != "EDIT_ARMATURE":
        # A hidden object cannot be the active object for a mode switch, so the
        # mode_set poll would raise "Context missing active object". Unhide the
        # armature first so it is visible and settable as the active object.
        for _armature_object in armature_object:
            _armature_object.hide_viewport = False
            set_hidden(_armature_object, False)
        select_only(*armature_object)
        if not set_active(armature_object[0] if armature_object else None):
            return
        bpy.ops.object.mode_set(mode="EDIT")


def switch_to_pose_mode(*scene_object: bpy.types.Object):
    switch_to_object_mode()
    # A hidden object cannot be the active object for a mode switch, so the
    # mode_set poll would raise "Context missing active object". Unhide the
    # objects first so they are visible and settable as the active object.
    for _scene_object in scene_object:
        _scene_object.hide_viewport = False
        set_hidden(_scene_object, False)
    if not select_only(*scene_object):
        return
    bpy.ops.object.mode_set(mode="POSE")


def apply_pose(rig_object: bpy.types.Object, selected: bool = False):
    switch_to_object_mode()
    switch_to_pose_mode(rig_object)
    bpy.ops.pose.armature_apply(selected=selected)


def apply_transforms(
    scene_object: bpy.types.Object,
    location: bool = False,
    rotation: bool = False,
    scale: bool = False,
    recursive: bool = False,
) -> None:
    deselect_all()
    switch_to_object_mode()
    select_only(scene_object)
    bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)

    if recursive:
        for child_object in scene_object.children:
            apply_transforms(child_object, location=location, rotation=rotation, scale=scale, recursive=recursive)


def walk_children(scene_object: bpy.types.Object) -> Generator[bpy.types.Object, None, None]:
    yield scene_object
    for child in scene_object.children:
        yield from walk_children(child)


def hide_empties():
    for scene_object in bpy.data.objects:
        if scene_object.name.startswith("GRP_"):
            scene_object.hide_viewport = True


def set_hide_recursively(scene_object: bpy.types.Object, value: bool) -> None:
    for child in walk_children(scene_object):
        child.hide_set(value)


def set_viewport_shading(mode: str) -> None:
    if not bpy.context.screen:
        return

    for area in bpy.context.screen.areas:
        if area.ui_type == "VIEW_3D":
            for space in area.spaces:
                if hasattr(space, "shading"):
                    space.shading.type = mode  # type: ignore[attr-defined]


def get_addon_version() -> str:
    addon_version = "unknown"
    blender_manifest = Path(__file__).parent.parent / "blender_manifest.toml"
    if blender_manifest.exists():
        with blender_manifest.open("rb") as f:
            data = tomllib.load(f)
            addon_version = data.get("version", addon_version)
    return addon_version


def notify_rig_instances_changed(instance: "RigInstance | None" = None) -> None:
    """Fire the registered post-setup callbacks after the rig instance list changes.

    Call this whenever a rig instance is added or removed at runtime (import, duplicate,
    delete) so external integrations -- e.g. the Character Control Rig addon -- can resync
    their own state. ``instance`` is the affected rig instance for additions, or ``None`` for
    removals (where the callback should simply re-derive its state from the current list).
    """
    from .. import post_setup_scene_callbacks
    from ..runtime.controller import request_sync

    request_sync()

    for callback in post_setup_scene_callbacks:
        try:
            callback(instance)
        except Exception as error:
            logger.exception(f"Error in post_setup_scene_callbacks: {error}")


def setup_scene(*_: Any) -> None:
    from ..runtime import controller, engine

    engine.invalidate()
    # Auto-migrate rig-instance data saved by a different addon edition (Free vs
    # Pro) or an older version before initializing, so reopening a .blend always
    # yields a correctly populated rig-instance list. Guard against failures so a
    # migration error never aborts scene setup.
    if detect_legacy_data(bpy.context.scene):
        try:
            migrate_legacy_data(bpy.context)
        except Exception as error:
            logger.exception(f"Failed to auto-migrate legacy scene data: {error}")

    scene_properties = getattr(bpy.context.scene, ToolInfo.NAME, object)

    # initialize the rig instances
    for instance in getattr(scene_properties, "rig_instance_list", []):
        try:
            # notify any registered callbacks that a rig instance has been set up in the scene, so they can perform
            # any necessary actions
            with controller.preserve_bindings():
                notify_rig_instances_changed(instance)
        except Exception as error:
            logger.exception(f"Failed to set up rig instance '{instance.name}': {error}")

    controller.after_load()


def teardown_scene(*_: Any) -> None:
    from ..runtime import controller, engine

    engine.invalidate()
    scene_properties = getattr(bpy.context.scene, ToolInfo.NAME, object)

    with controller.preserve_bindings():
        for instance in getattr(scene_properties, "rig_instance_list", []):
            instance.destroy()
    logger.info("De-allocated Rig Logic instances...")


def pre_undo(*_: Any) -> None:
    from ..runtime.controller import before_undo

    before_undo()
    context: "Context" = bpy.context  # type: ignore[attr-defined]  # noqa: UP037
    addon_window_manager_properties = get_addon_window_manager_properties(context)
    addon_scene_properties = get_addon_scene_properties(context)

    # Always invalidate the caches that hold live `bpy` RNA wrappers
    for instance in addon_scene_properties.rig_instance_list:
        instance.destroy_references()

    # Only run the pre-undo logic if the current context is a 3D view area
    if context.area and context.area.type == "VIEW_3D" and context.region and context.region.type == "WINDOW":
        addon_window_manager_properties.evaluate_dependency_graph = False
        addon_window_manager_properties.is_undoing = True
        active_object = bpy.context.active_object
        # destroy cached data related rig instances, since undo can change the data
        # in a way that makes the cached data invalid
        for instance in addon_scene_properties.rig_instance_list:
            if (
                active_object in [instance.head_mesh, instance.head_rig, instance.face_board]
                and instance.auto_evaluate_head
            ):
                instance.destroy_head()
            if (
                active_object in [instance.body_mesh, instance.body_rig, instance.control_rig]
                and instance.auto_evaluate_body
            ):
                instance.destroy_body()


def post_undo(*_: Any) -> None:
    from ..runtime.controller import after_undo

    after_undo()
    context: "Context" = bpy.context  # type: ignore[attr-defined]  # noqa: UP037
    addon_window_manager_properties = get_addon_window_manager_properties(context)

    # Only run the post-undo logic if the current context is a 3D view area
    if context.area and context.area.type == "VIEW_3D" and context.region and context.region.type == "WINDOW":
        addon_window_manager_properties.evaluate_dependency_graph = True


def pre_redo(*args: Any) -> None:
    pre_undo(*args)


def post_redo(*args: Any) -> None:
    post_undo(*args)


def pre_render(*_: Any) -> None:
    begin_render()


def post_render(*_: Any) -> None:
    end_render()


def post_save(*_: Any) -> None:
    instance = get_active_rig_instance()
    if not instance:
        return

    # Create a DNA backup (Pro editors only; no-op in the free edition).
    try:
        from ..editors.backup_manager.core import BackupType, create_backup
    except ImportError:
        return

    create_backup(instance, BackupType.BLENDER_FILE_SAVE)


def create_empty(empty_name: str) -> bpy.types.Object:
    empty_object = bpy.data.objects.get(empty_name)
    if not empty_object:
        empty_object = bpy.data.objects.new(empty_name, object_data=None)

    if bpy.context.scene and empty_object not in bpy.context.scene.collection.objects.values():
        bpy.context.scene.collection.objects.link(empty_object)

    return empty_object


def toggle_expand_in_outliner(state: int = 2):
    """
    Collapses or expands the collections in any outliner region on the current screen.


    Args:
        state (int, optional): 1 will expand all collections, 2 will
            collapse them. Defaults to 2.
    """
    if not bpy.context.screen:
        return
    for area in bpy.context.screen.areas:
        if area.type == "OUTLINER":
            for region in area.regions:
                if region.type == "WINDOW":
                    with bpy.context.temp_override(area=area, region=region):  # type: ignore[arg-type]
                        bpy.ops.outliner.show_hierarchy()
                        for _i in range(state):
                            bpy.ops.outliner.expanded_toggle()
                    area.tag_redraw()


def focus_on_selected():
    """
    Focuses any 3D view region on the current screen to the selected object.
    """
    if not bpy.context.screen or not bpy.context.window_manager:
        return
    for window in bpy.context.window_manager.windows:
        if window.screen:
            for area in bpy.context.screen.areas:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            with bpy.context.temp_override(area=area, region=region):  # type: ignore[arg-type]
                                bpy.ops.view3d.view_selected()


def remove_instance_prefix(name: str, instance_name: str) -> str:
    """Remove one leading rig-instance namespace from a scene data-block name."""
    if not instance_name:
        return name
    return name.removeprefix(f"{instance_name}_")


def replace_instance_prefix(name: str, old_instance_name: str, new_instance_name: str) -> str:
    """Replace an exact or leading rig-instance namespace in a data-block name."""
    if name == old_instance_name:
        return new_instance_name
    prefix = f"{old_instance_name}_"
    if name.startswith(prefix):
        return f"{new_instance_name}_{name.removeprefix(prefix)}"
    return name


def get_head(name: str) -> "CharacterComponentHead | None":
    # avoid circular import
    from ..components.head import CharacterComponentHead

    scene_properties = get_addon_scene_properties()
    for instance in scene_properties.rig_instance_list:
        if instance.name == name:
            return CharacterComponentHead(rig_instance=instance, component_type="head")

    logger.error(f'No existing head "{name}" was found')
    return None


def get_body(name: str) -> "CharacterComponentBody | None":
    # avoid circular import
    from ..components.body import CharacterComponentBody

    scene_properties = get_addon_scene_properties()
    for instance in scene_properties.rig_instance_list:
        if instance.name == name:
            return CharacterComponentBody(rig_instance=instance, component_type="body")

    logger.error(f'No existing body "{name}" was found')
    return None


def get_active_head() -> "CharacterComponentHead | None":
    """
    Gets the active head object.
    """
    instance = get_active_rig_instance()
    if instance:
        return get_head(instance.name)
    return None


def get_active_body() -> "CharacterComponentBody | None":
    """
    Gets the active body object.
    """
    instance = get_active_rig_instance()
    if instance:
        return get_body(instance.name)
    return None


def is_collection_in_scene(collection: bpy.types.Collection | None) -> bool:
    """Whether the collection is reachable from the scene's root collection."""
    if not collection or not bpy.context.scene:
        return False
    root = bpy.context.scene.collection
    return collection == root or collection in root.children_recursive


def move_to_collection(scene_objects: list[bpy.types.Object], collection_name: str, exclusively: bool = False):
    collection = bpy.data.collections.get(collection_name)
    if not collection and bpy.context.scene:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    # A collection of this name can survive in the file without being linked to the scene, and
    # moving objects into it would drop them out of the view layer entirely.
    if collection and not is_collection_in_scene(collection) and bpy.context.scene:
        logger.warning(f'Re-linking collection "{collection.name}" to the scene, it was outside the scene hierarchy.')
        bpy.context.scene.collection.children.link(collection)

    if exclusively:
        # unlink the objects from their current collections
        for scene_object in scene_objects:
            for user_collection in scene_object.users_collection:
                user_collection.objects.unlink(scene_object)

    # link the objects to the new collection
    for scene_object in scene_objects:
        if collection and scene_object not in collection.objects.values():
            collection.objects.link(scene_object)


def group_face_board_with_linked_collection(
    face_board: bpy.types.Object,
    linked_collection: bpy.types.Collection,
    collection_name: str,
) -> None:
    """Group a local face board together with a library-linked character collection.

    A local object cannot be linked into a library-linked collection, so a local wrapper
    collection of the same name is created instead and the linked collection is nested
    under it alongside the face board. Collection names are namespaced per library, so the
    local wrapper keeps ``collection_name`` rather than gaining a ``.001`` suffix.
    """
    scene = bpy.context.scene
    if not scene:
        return

    wrapper = bpy.data.collections.new(collection_name)
    scene.collection.children.link(wrapper)

    if any(child == linked_collection for child in scene.collection.children):
        scene.collection.children.unlink(linked_collection)
    wrapper.children.link(linked_collection)

    for user_collection in face_board.users_collection:
        user_collection.objects.unlink(face_board)
    wrapper.objects.link(face_board)


def set_origin_to_world_center(scene_object: bpy.types.Object):
    switch_to_object_mode()
    # set the active object
    select_only(scene_object)
    # snap the cursor to the world center
    bpy.ops.view3d.snap_cursor_to_center()
    # then move the origin to match the cursor
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="BOUNDS")


def set_objects_origins(scene_objects: list[bpy.types.Object], location: Vector):
    if not bpy.context.scene:
        return

    switch_to_object_mode()
    # set the active object
    for scene_object in scene_objects:
        select_only(scene_object)
        # snap the cursor to the world center
        bpy.context.scene.cursor.location = location
        # then move the origin to match the cursor
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="BOUNDS")
        apply_transforms(scene_object, location=True, rotation=True, scale=True)


def rename_rig_instance(instance: "RigInstance", old_name: str, new_name: str):  # noqa: PLR0912
    from ..runtime import controller, engine

    if instance.face_board:
        instance.face_board.name = replace_instance_prefix(instance.face_board.name, old_name, new_name)
        instance.face_board.data.name = replace_instance_prefix(instance.face_board.data.name, old_name, new_name)
    if instance.head_mesh:
        instance.head_mesh.name = replace_instance_prefix(instance.head_mesh.name, old_name, new_name)
        instance.head_mesh.data.name = replace_instance_prefix(instance.head_mesh.data.name, old_name, new_name)
    if instance.head_rig:
        instance.head_rig.name = replace_instance_prefix(instance.head_rig.name, old_name, new_name)
        instance.head_rig.data.name = replace_instance_prefix(instance.head_rig.data.name, old_name, new_name)
    if instance.head_material:
        instance.head_material.name = replace_instance_prefix(instance.head_material.name, old_name, new_name)
    if instance.body_mesh:
        instance.body_mesh.name = replace_instance_prefix(instance.body_mesh.name, old_name, new_name)
        instance.body_mesh.data.name = replace_instance_prefix(instance.body_mesh.data.name, old_name, new_name)
    if instance.body_rig:
        instance.body_rig.name = replace_instance_prefix(instance.body_rig.name, old_name, new_name)
        instance.body_rig.data.name = replace_instance_prefix(instance.body_rig.data.name, old_name, new_name)
    if instance.control_rig:
        instance.control_rig.name = replace_instance_prefix(instance.control_rig.name, old_name, new_name)
        instance.control_rig.data.name = replace_instance_prefix(instance.control_rig.data.name, old_name, new_name)
    if instance.body_material:
        instance.body_material.name = replace_instance_prefix(instance.body_material.name, old_name, new_name)

    for item in instance.output.head_item_list.values() + instance.output.body_item_list.values():
        # don't rename these again
        if item.scene_object in [
            instance.face_board,
            instance.head_mesh,
            instance.head_rig,
            instance.body_mesh,
            instance.body_rig,
            instance.control_rig,
        ]:
            continue

        if item.scene_object:
            item.scene_object.name = replace_instance_prefix(item.scene_object.name, old_name, new_name)
            item.scene_object.data.name = replace_instance_prefix(item.scene_object.data.name, old_name, new_name)
        if item.image_object:
            item.image_object.name = replace_instance_prefix(item.image_object.name, old_name, new_name)

    # rename the main collection
    main_collection = bpy.data.collections.get(old_name)
    if main_collection:
        main_collection.name = new_name

    # rename the LOD collections
    for index in range(NUMBER_OF_HEAD_LODS):
        collection = bpy.data.collections.get(f"{old_name}_lod{index}")
        if collection:
            collection.name = replace_instance_prefix(collection.name, old_name, new_name)

    # this frees up the instance data under the old name, since all data is
    # namespaced under the instance name
    with controller.preserve_bindings():
        instance.destroy()
    for carrier in engine.carriers(instance):
        if not carrier.library and not carrier.override_library:
            carrier["character_name"] = new_name
            carrier.name = replace_instance_prefix(carrier.name, old_name, new_name)
    drivers = bpy.data.collections.get((f"{old_name}_drivers", None))
    if drivers:
        drivers.name = f"{new_name}_drivers"


def rename_as_lod0_meshes(mesh_objects: list[bpy.types.Object]):
    from ..ui.callbacks import update_head_output_items

    instance = get_active_rig_instance()
    if instance:
        for mesh_object in mesh_objects:
            mesh_object.name = re.sub(INVALID_NAME_CHARACTERS_REGEX, "_", mesh_object.name.strip())
            if not mesh_object.name.startswith(instance.name):
                mesh_object.name = f"{instance.name}_{mesh_object.name}"
            if not mesh_object.name.endswith("_lod0_mesh"):
                mesh_object.name = f"{mesh_object.name}_lod0_mesh"

        # re-populate the output items
        instance.output.head_item_list.clear()
        update_head_output_items(None, bpy.context)  # type: ignore[arg-type]


def report_error(message: str):
    """
    Raises and error pop up to report a error message to the user.

    Args:
        message (str): The body text with the error message.
    """
    ops = get_addon_ops_module()
    ops.report_error(
        # "INVOKE_DEFAULT",
        message=message
    )


def report_error_panel(title: str, message: str, fix: Callable | None = None, width: int = 500):
    """
    Raises and error dialog to report error messages to the user with an optional fix.

    Args:
        title (str): The title of the error in the modal header.

        message (str): The body text with the error message.

        fix (Callable | None, optional): An optional function to be run to
            fix the issue if the user confirms. Defaults to None.

        width (int, optional): The width of the modal. Defaults to 500.
    """
    addon_window_manager_properties = get_addon_window_manager_properties()
    addon_window_manager_properties.errors[title] = {"fix": fix}
    ops = get_addon_ops_module()
    ops.report_error_with_fix(
        "INVOKE_DEFAULT",
        title=title,
        message=message,
        width=width,
    )


def import_head_texture_logic_node() -> bpy.types.NodeTree | None:
    sep = "\\"
    if sys.platform != "win32":
        sep = "/"

    node_group = bpy.data.node_groups.get(HEAD_TEXTURE_LOGIC_NODE_LABEL)
    if not node_group:
        directory_path = f"{MATERIALS_FILE_PATH}{sep}NodeTree{sep}"
        file_path = f"{MATERIALS_FILE_PATH}{sep}NodeTree{sep}{HEAD_TEXTURE_LOGIC_NODE_LABEL}"
        bpy.ops.wm.append(filepath=file_path, filename=HEAD_TEXTURE_LOGIC_NODE_LABEL, directory=directory_path)
        return bpy.data.node_groups.get(HEAD_TEXTURE_LOGIC_NODE_LABEL)
    return node_group


def dependencies_are_valid() -> bool:
    """Require both the DNA authoring bindings and the native evaluator."""
    try:
        from ..bindings import dna, load_native_runtime, riglogic

        load_native_runtime()
    except Exception:
        return False
    return not (getattr(dna, "__is_fake__", False) or getattr(riglogic, "__is_fake__", False))


def reduce_close_floats(float_list: list[float], tolerance: float = DEFAULT_UV_TOLERANCE) -> list[float]:
    """
    Reduces a list of floats by removing values that are too close to each other.

    Args:
        float_list: The list of floats to reduce.
        tolerance: The maximum allowed difference for two floats to be considered "close".

    Returns:
        A new list with close values reduced.
    """
    if not float_list:
        return []

    sorted_list = sorted(set(float_list))  # Sort and remove exact duplicates first
    if not sorted_list:
        return []

    reduced_list = [sorted_list[0]]
    for i in range(1, len(sorted_list)):
        # Compare with the last added element in the reduced_list
        if not math.isclose(sorted_list[i], reduced_list[-1], abs_tol=tolerance):
            reduced_list.append(sorted_list[i])
    return reduced_list


def shell(command: str, **kwargs: Any) -> Generator[str, None, None]:
    """
    Runs the command is a fully qualified shell.

    Args:
        command (str): A command.

    Yields:
        str: The output of the command line by line.

    Raises:
        OSError: The error cause by the shell.
    """
    process = subprocess.Popen(  # noqa: S602
        command, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
    )

    output = []
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            output += [line.rstrip()]
            yield line.rstrip()

    process.wait()

    if process.returncode != 0:
        raise OSError("\n".join(output))


def add_rig_instance(name: None | str = None) -> "RigInstance":
    scene_properties = get_addon_scene_properties()
    my_list = scene_properties.rig_instance_list
    active_index = scene_properties.rig_instance_list_active_index
    to_index = min(len(my_list), active_index + 1)
    instance: "RigInstance" = my_list.add()  # noqa: UP037

    if not name:
        instance.name = f"Untitled{len(my_list)}"
    else:
        instance.name = name

    my_list.move(len(my_list) - 1, to_index)
    scene_properties.rig_instance_list_active_index = to_index
    return instance


def extract_rig_instance_data_from_blend_file(blend_file_path: Path) -> tuple[list[dict], str]:
    extracted_data = []

    file_id = uuid.uuid4()
    script_file = SCRIPTS_FOLDER / "save_rig_instance_data.py"
    data_file = TEMP_FOLDER / f"{file_id}.json"
    error_file = TEMP_FOLDER / f"{file_id}_error.log"
    addon_folder = Path(__file__).parent.parent.parent

    binary_path = bpy.app.binary_path
    if binary_path:
        if sys.platform == "win32":
            command = (
                f'"{binary_path}" --background --python "{script_file}" -- --data-file "{data_file}" '
                f'--blend-file "{blend_file_path}" --addon-folder "{addon_folder}" --addon-name "{ToolInfo.NAME}"'
            )
        else:
            command = (
                f"{binary_path} --background --python {script_file.as_posix()} -- --data-file {data_file.as_posix()} "
                f"--blend-file {blend_file_path.as_posix()} --addon-folder {addon_folder.as_posix()} "
                f"--addon-name {ToolInfo.NAME}"
            )
    # binary path can be empty if blender is run headless
    elif sys.platform == "win32":
        command = (
            f'"{sys.executable}" "{script_file}" -- --data-file "{data_file}" --blend-file "{blend_file_path}" '
            f'--addon-folder "{addon_folder}" --addon-name "{ToolInfo.NAME}"'
        )
    else:
        command = (
            f"{sys.executable} {script_file.as_posix()} -- --data-file {data_file.as_posix()} --blend-file "
            f"{blend_file_path.as_posix()} --addon-folder {addon_folder.as_posix()} --addon-name {ToolInfo.NAME}"
        )

    # Run the extraction in a headless Blender subprocess. The bpy module is known to
    # segfault during interpreter teardown, which yields a non-zero exit code even when
    # the extraction itself succeeded. The authoritative results are the error log and
    # data file written before teardown, so capture the output and always inspect those
    # files instead of failing the moment the subprocess returns non-zero.
    subprocess_output: list[str] = []
    try:
        for line in shell(command=command):
            subprocess_output.append(line)  # noqa: PERF402
    except OSError as error:
        if not subprocess_output:
            subprocess_output = str(error).splitlines()
        logger.debug("Rig instance extraction subprocess exited non-zero: %s", error)

    if error_file.exists():
        with error_file.open() as f:
            error_message = f.read()

        try:
            error_file.unlink()
        except OSError as error:
            logger.debug(error)

        return [], error_message

    if data_file.exists():
        with data_file.open() as f:
            extracted_data = json.load(f)

        try:
            data_file.unlink()
        except OSError as error:
            logger.debug(error)

        return extracted_data, ""

    # Neither a data file nor an error log was produced, so surface the full subprocess
    # output to make the failure actionable instead of reporting a generic message.
    if subprocess_output:
        return [], "Failed to extract rig instance data from blend file:\n" + "\n".join(subprocess_output)

    return [], "Failed to extract rig instance data."


def reset_face_board_controls(face_board_object: bpy.types.Object) -> None:
    """Return a face board's expression controls to neutral.

    Only leaf ``CTRL_`` bones are moved, so the panel's layout and switch bones keep the
    positions that define the board itself.

    Args:
        face_board_object: The face board armature object.
    """
    if not face_board_object.pose:
        return

    for pose_bone in face_board_object.pose.bones:
        if (
            not pose_bone.bone.children
            and pose_bone.name.startswith("CTRL_")
            and pose_bone.name not in FACE_BOARD_SWITCHES + EXCLUDED_FACE_BOARD_CONTROLS
        ):
            pose_bone.location = Vector((0.0, 0.0, 0.0))


def duplicate_face_board(name: str) -> bpy.types.Object | None:
    scene_properties = get_addon_scene_properties()
    for instance in scene_properties.rig_instance_list:
        if instance.face_board:
            # Duplicate the face board object
            face_board_duplicate = instance.face_board.copy()
            face_board_duplicate.name = f"{name}_{FACE_BOARD_NAME}"
            face_board_duplicate.data = instance.face_board.data.copy()
            face_board_duplicate.data.name = f"{name}_{FACE_BOARD_NAME}"
            # The copy carries the source character's animation and expression, which would
            # otherwise drive the new character's face and skew where the board is placed.
            face_board_duplicate.animation_data_clear()
            # Bone visibility drivers live on the armature data in Blender 4.5.
            face_board_duplicate.data.animation_data_clear()
            reset_face_board_controls(face_board_duplicate)
            if bpy.context.collection:
                bpy.context.collection.objects.link(face_board_duplicate)
            return face_board_duplicate
    return None


def hide_face_board_widgets():
    # unlink from scene and make fake users so they are not deleted by garbage collection
    for empty_name in FACE_GUI_EMPTIES:
        empty = bpy.data.objects.get(empty_name)
        if empty and bpy.context.scene:
            for collection in [
                bpy.data.collections.get("Collection"),
                bpy.context.scene.collection,
            ]:
                if not collection:
                    continue

                for child in empty.children_recursive:
                    if child in collection.objects.values():
                        collection.objects.unlink(child)
                    child.use_fake_user = True

                if empty in collection.objects.values():
                    collection.objects.unlink(empty)
                empty.use_fake_user = True


def purge_face_board_components():
    with bpy.data.libraries.load(str(FACE_BOARD_FILE_PATH)) as (data_from, _data_to):  # type: ignore[arg-type]
        if data_from.objects:
            for name in data_from.objects:
                scene_object = bpy.data.objects.get(name)
                if scene_object:
                    bpy.data.objects.remove(scene_object, do_unlink=True)


def import_face_board(name: str) -> bpy.types.Object | None:
    sep = "\\"
    if sys.platform != "win32":
        sep = "/"

    # delete all face board objects in the scene that already exist
    purge_face_board_components()

    bpy.ops.wm.append(
        filepath=f"{FACE_BOARD_FILE_PATH}{sep}Object{sep}{FACE_BOARD_NAME}",
        filename=FACE_BOARD_NAME,
        directory=f"{FACE_BOARD_FILE_PATH}{sep}Object{sep}",
    )
    face_board_object = bpy.data.objects[FACE_BOARD_NAME]
    # rename to be prefixed with a unique name
    face_board_object.name = f"{name}_{FACE_BOARD_NAME}"

    # hide all face board elements
    hide_face_board_widgets()

    if isinstance(face_board_object.data, bpy.types.Armature):
        face_board_object.data.relation_line_position = "HEAD"
    return face_board_object


def set_armature_object_origin(armature_object: bpy.types.Object, world_location: Vector) -> None:
    """Move an armature object's origin to a world location without moving its bones.

    Only the world translation changes, so bone rolls, pose transforms and the
    resulting deformation are all preserved.

    Args:
        armature_object: The armature object to re-origin.
        world_location: The new world-space location of the object origin.
    """
    if not isinstance(armature_object.data, bpy.types.Armature):
        return

    previous_matrix = armature_object.matrix_world.copy()
    new_matrix = previous_matrix.copy()
    new_matrix.translation = world_location
    to_new_local = new_matrix.inverted()

    switch_to_object_mode()
    switch_to_bone_edit_mode(armature_object)
    # Snapshot every bone in world space first. Writing a connected bone's head also moves
    # its parent's tail, so reading and writing in the same pass would transform bones twice.
    world_positions = {
        edit_bone.name: (previous_matrix @ edit_bone.head.copy(), previous_matrix @ edit_bone.tail.copy())
        for edit_bone in armature_object.data.edit_bones
    }
    for edit_bone in armature_object.data.edit_bones:
        head, tail = world_positions[edit_bone.name]
        edit_bone.head = to_new_local @ head
        edit_bone.tail = to_new_local @ tail
    switch_to_object_mode()

    armature_object.matrix_world = new_matrix


def reset_child_of_inverses(armature_object: bpy.types.Object) -> None:
    """Clear and re-set the inverse matrix on every Child Of constraint of an armature.

    The stored inverse is relative to the owner object's transform, so it becomes
    stale whenever the owner's origin moves.

    Args:
        armature_object: The armature object whose pose bone constraints are refreshed.
    """
    if not armature_object.pose:
        return

    switch_to_pose_mode(armature_object)
    for pose_bone in armature_object.pose.bones:
        for constraint in pose_bone.constraints:
            if constraint.type != "CHILD_OF":
                continue
            with bpy.context.temp_override(active_object=armature_object, active_pose_bone=pose_bone):  # type: ignore[arg-type]
                bpy.ops.constraint.childof_clear_inverse(constraint=constraint.name, owner="BONE")
                bpy.ops.constraint.childof_set_inverse(constraint=constraint.name, owner="BONE")


@preserve_context
def align_face_board_origin(face_board_object: bpy.types.Object, rig_object: bpy.types.Object) -> bool:
    """Match a face board's origin to the origin of the rig it belongs to.

    A duplicated face board inherits the origin of the instance it was copied from, so in a
    multi-character scene its origin no longer sits on the new character's body rig. A face
    board shared by more than one rig instance is left untouched.

    Args:
        face_board_object: The face board armature object.
        rig_object: The rig instance's body rig (or head rig when there is no body).

    Returns:
        True when the origin was moved.
    """
    if not face_board_object or not rig_object:
        return False

    scene_properties = get_addon_scene_properties()
    if scene_properties:
        users = [i for i in scene_properties.rig_instance_list if i.face_board == face_board_object]
        if len(users) > 1:
            logger.debug(f'Face board "{face_board_object.name}" is shared by multiple rig instances. Skipping.')
            return False

    world_location = rig_object.matrix_world.translation.copy()
    if (face_board_object.matrix_world.translation - world_location).length <= _ORIGIN_TOLERANCE:
        return False

    set_armature_object_origin(face_board_object, world_location)
    reset_child_of_inverses(face_board_object)
    logger.info(f'Moved face board "{face_board_object.name}" origin to match "{rig_object.name}".')
    return True


def un_constrain_face_board_to_head(face_board_object: bpy.types.Object, bone_name: str) -> None:
    if face_board_object and face_board_object.pose:
        switch_to_pose_mode(face_board_object)
        pose_bone = face_board_object.pose.bones.get(bone_name)
        if pose_bone:
            for constraint in pose_bone.constraints:
                if constraint.type == "CHILD_OF":
                    pose_bone.constraints.remove(constraint)


def constrain_face_board_to_head(
    head_rig_object: bpy.types.Object,
    body_rig_object: bpy.types.Object,
    face_board_object: bpy.types.Object,
    bone_name: str,
) -> None:
    if head_rig_object and face_board_object and face_board_object.pose:
        switch_to_pose_mode(face_board_object)
        pose_bone = face_board_object.pose.bones.get(bone_name)
        if pose_bone:
            constraint = None
            for existing_constraint in pose_bone.constraints:
                if existing_constraint.type == "CHILD_OF":
                    constraint = existing_constraint
                    break
            if not constraint:
                constraint = pose_bone.constraints.new(type="CHILD_OF")

            rig_object = body_rig_object or head_rig_object
            constraint.target = rig_object  # type: ignore[attr-defined]
            constraint.subtarget = "head"  # type: ignore[attr-defined]
            # Set the inverse matrix using the operator
            with bpy.context.temp_override(active_object=face_board_object, active_pose_bone=pose_bone):  # type: ignore[arg-type]
                bpy.ops.constraint.childof_set_inverse(constraint=constraint.name, owner="BONE")


@preserve_context
def position_eye_aim(head_rig_object: bpy.types.Object, face_board_object: bpy.types.Object) -> None:
    if head_rig_object and face_board_object and face_board_object.pose and head_rig_object.pose:
        un_constrain_face_board_to_head(face_board_object, bone_name="CTRL_C_eyesAim")

        left_eye_bone = head_rig_object.pose.bones.get("FACIAL_L_Eye")
        right_eye_bone = head_rig_object.pose.bones.get("FACIAL_R_Eye")
        if left_eye_bone and right_eye_bone:
            eye_center = head_rig_object.matrix_world.inverted() @ ((left_eye_bone.head + right_eye_bone.head) / 2)
            target_eye_aim_world_location = eye_center + Vector((0, -0.3, 0))

            switch_to_edit_mode(face_board_object)
            if isinstance(face_board_object.data, bpy.types.Armature):
                eye_aim_center = face_board_object.data.edit_bones.get("CTRL_C_eyesAim")
                if eye_aim_center:
                    eye_aim_world_location = face_board_object.matrix_world.inverted() @ eye_aim_center.head

                    # calculate the offset between the current eye aim location and the target location
                    offset = eye_aim_world_location - target_eye_aim_world_location

                    # move all eye aim bones by the offset
                    for bone_name in EYE_AIM_BONES:
                        bone = face_board_object.data.edit_bones.get(bone_name)
                        if bone:
                            bone.head -= offset
                            bone.tail -= offset


def position_face_board(
    head_mesh_object: bpy.types.Object | None,
    head_rig_object: bpy.types.Object | None,
    face_board_object: bpy.types.Object,
) -> None:
    from .mesh import get_bounding_box_center, get_bounding_box_left_x, get_bounding_box_right_x

    if head_mesh_object and head_rig_object:
        un_constrain_face_board_to_head(face_board_object, bone_name="CTRL_faceGUI")
        un_constrain_face_board_to_head(face_board_object, bone_name="CTRL_C_eyesAim")
        # Bounding boxes are read from the evaluated pose, so the dropped constraints have to
        # be flushed first. A duplicated board is otherwise still measured where it sat on the
        # character it was copied from.
        if bpy.context.view_layer:
            bpy.context.view_layer.update()

        head_mesh_center = get_bounding_box_center(head_mesh_object)
        face_gui_center = get_bounding_box_center(face_board_object)
        head_mesh_right_x = get_bounding_box_right_x(head_mesh_object)
        face_gui_left_x = get_bounding_box_left_x(face_board_object)

        # align the face gui object to the head mesh vertically
        translation_vector = head_mesh_center - face_gui_center
        face_board_object.location.z += translation_vector.z

        # offset the face gui object to the left of the head mesh
        face_board_object.location.x += head_mesh_right_x - face_gui_left_x

        # apply the translation to the face gui object
        apply_transforms(face_board_object, location=True)

        # position the eye aim controls
        position_eye_aim(head_rig_object, face_board_object)


def collection_to_list(collection: bpy.types.bpy_prop_collection) -> list:
    item_list = []
    for item in collection:
        data = {"__property_group__": item.__class__.__name__}
        for key, data_type in item.__annotations__.items():
            if data_type.function.__name__ == "CollectionProperty":
                data[key] = collection_to_list(getattr(item, key))
            elif data_type.function.__name__ == "FloatVectorProperty":
                data[key] = getattr(item, key)[:]
            else:
                data[key] = getattr(item, key)

        item_list.append(data)
    return item_list


def get_raw_scene_data(scene: bpy.types.Scene, addon_id: str) -> Any:
    """Return the rig-instance data group stored under ``addon_id`` on ``scene``.

    The current edition and its read-only sibling (Free vs Pro) are registered
    scene ``PointerProperty`` groups, so their data is exposed via attribute
    access. The old ``meta_human_dna`` prototype instead stored plain custom
    properties, which are read via subscript. ID-pointer sub-properties resolve to
    their real datablocks when read from the returned group either way. Returns
    ``None`` when no data is present.
    """
    if not scene:
        return None
    group = getattr(scene, addon_id, None)
    if group is not None:
        return group
    try:
        return scene.get(addon_id)
    except (KeyError, TypeError):
        return None


def _field(source: Any, name: str) -> Any:
    """Read ``name`` from a rig-instance ``source``.

    ``source`` is either a live ``RigInstance`` (a registered edition, read via
    RNA attribute) or a raw IDProperty group/dict from the old prototype (read via
    ``get``). A live RNA struct exposes ``bl_rna``; a raw IDProperty group does
    not.
    """
    if hasattr(source, "bl_rna"):
        return getattr(source, name, None)
    if hasattr(source, "get"):
        return source.get(name)
    return None


def _rig_instance_sources(group: Any, key: str) -> list:
    """Return the list of rig-instance sources held under ``key`` on ``group``."""
    if group is None:
        return []
    # Registered editions expose the list as an RNA collection, but only when that
    # edition defines this key -- the old prototype's registered group has no
    # `rig_instance_list`, so reading it unguarded raises AttributeError. Anything not
    # exposed through RNA was stored as a plain custom property.
    data = getattr(group, key, None) if hasattr(group, "bl_rna") else None
    if data is None and hasattr(group, "get"):
        data = group.get(key)
    return list(data) if data else []


def detect_legacy_data(scene: bpy.types.Scene) -> tuple[str, str] | None:
    """Detect rig-instance data that belongs to a different edition or version.

    Returns a ``(addon_id, data_key)`` tuple for the first scene key that holds
    migratable rig-instance data, or ``None`` when nothing needs migrating. A key
    qualifies when it is a *foreign* edition (not :attr:`ToolInfo.NAME`) holding
    rig-instance data, or when the current edition still stores the old
    ``rig_logic_instance_list`` format that must be upgraded in place. ``data_key``
    is an empty string when only asset collections survived (collection-data
    migration).
    """
    if not scene:
        return None

    for addon_id in ADDON_IDS:
        group = get_raw_scene_data(scene, addon_id)
        if group is None:
            continue

        for key in MIGRATABLE_DATA_KEYS:
            if _rig_instance_sources(group, key) and (addon_id != ToolInfo.NAME or key in LEGACY_DATA_KEYS):
                return addon_id, key

        # An old-prototype custom-property group with no rig-instance list can
        # still be reconstructed from the asset collection names in the scene.
        if addon_id != ToolInfo.NAME and not hasattr(group, "bl_rna") and not group:
            return addon_id, ""

    return None


def detect_runtime_migration(scene: bpy.types.Scene) -> bool:
    """Detect outdated saved bindings, excluding outputs temporarily owned by editors."""
    from ..runtime import controller, engine

    properties = getattr(scene, ToolInfo.NAME, None)
    for instance in getattr(properties, "rig_instance_list", ()):
        if controller.is_suspended(instance):
            continue
        if _runtime_migration_components(instance) and engine.binding_issues(instance):
            return True
    return False


def migrate_runtime_data(context: "Context") -> tuple[str, int, int]:
    """Preflight the batch, convert metadata, and rebuild only outdated local rigs.

    Return the metadata conversion kind, upgraded count, and unchanged count.
    """
    from ..runtime import controller, engine

    properties = get_addon_scene_properties(context)
    instances = list(properties.rig_instance_list)
    editing = [instance.name for instance in instances if controller.is_suspended(instance)]
    if editing:
        raise ValueError(f"Commit or revert active edits before migrating: {', '.join(editing)}")
    targets = [
        instance
        for instance in instances
        if _runtime_migration_components(instance) and engine.binding_issues(instance)
    ]
    sources = list(targets)
    detected = detect_legacy_data(context.scene)
    if detected:
        addon_id, data_key = detected
        existing_names = {instance.name for instance in instances}
        if data_key:
            sources.extend(
                source
                for source in _rig_instance_sources(get_raw_scene_data(context.scene, addon_id), data_key)
                if (_field(source, "name") or _field(source, "instance_name")) not in existing_names
            )
        else:
            sources.extend(
                {
                    "name": collection.name[:-5],
                    "head_rig": bpy.data.objects.get(collection.name[:-5] + "_head_rig"),
                    "body_rig": bpy.data.objects.get(collection.name[:-5] + "_body_rig"),
                }
                for collection in context.scene.collection.children_recursive
                if collection.name.endswith("_lod0") and collection.name[:-5] not in existing_names
            )
    eligible = [source for source in sources if _runtime_migration_components(source)]
    _preflight_runtime_migration(eligible)
    if eligible:
        available, reason = engine.capability()
        if not available:
            raise RuntimeError(reason)
    migrate_type = migrate_legacy_data(context)
    upgraded = 0
    unchanged = 0
    for instance in properties.rig_instance_list:
        if not _runtime_migration_components(instance) or not engine.binding_issues(instance):
            unchanged += 1
            continue
        controller.rebuild(instance)
        issues = engine.binding_issues(instance)
        if issues:
            raise RuntimeError(f"{instance.name}: runtime migration incomplete: {'; '.join(issues)}")
        upgraded += 1
    return migrate_type, upgraded, unchanged


def _preflight_runtime_migration(instances: list[Any]) -> None:  # noqa: PLR0912
    from ..runtime import bindings, engine

    issues = []
    for instance in instances:
        name = _field(instance, "name") or _field(instance, "instance_name")
        carriers = engine.carriers(instance)
        for carrier in carriers:
            version = carrier.get("schema_version", 1)
            if not isinstance(version, int) or version > engine.SCHEMA_VERSION:
                issues.append(f"{name}: unsupported future runtime schema {version}; update Character DNA")
        owners = list(carriers)
        scene = getattr(instance, "id_data", None)
        if scene is not None:
            owners.append(scene)
        for field in ("head_rig", "body_rig", "face_board", "control_rig", "head_mesh", "body_mesh"):
            owner = _resolve_datablock(_field(instance, field), bpy.data.objects)
            if isinstance(owner, bpy.types.Object):
                owners.extend((owner, owner.data))
        for carrier in carriers:
            for target in carrier.get("targets", ()):
                owner = target.get("owner")
                if owner is not None and target.get("embedded", False):
                    owner = owner.node_tree
                if owner is None:
                    continue
                owners.append(owner)
                animation = owner.animation_data
                path, index = target["path"], max(0, target["index"])
                curve = animation.drivers.find(path, index=index) if animation else None
                if curve and not bindings.owned_curve(curve, carrier):
                    issues.append(f"{name}: conflicting driver on {owner.name}:{path}[{index}]")
                if (path, index) in bindings._animated_channels(owner):  # noqa: SLF001
                    issues.append(f"{name}: keyframed native output on {owner.name}:{path}[{index}]")
        if any(
            owner is not None
            and (
                owner.library
                or not owner.is_editable
                or (owner.override_library and owner.override_library.is_system_override)
            )
            for owner in owners
        ):
            issues.append(f"{name}: linked or protected data; open the source .blend, migrate there, and save it")
        for component in _runtime_migration_components(instance):
            path = _field(instance, f"{component}_dna_file_path")
            if not path or not Path(bpy.path.abspath(path)).is_file():
                issues.append(
                    f"{name}: missing {component} DNA file path ({path or 'not set'}); assign an existing .dna"
                )
    if issues:
        raise ValueError("Cannot migrate rig runtime: " + "; ".join(issues))


def _runtime_migration_components(instance: Any) -> list[str]:
    return [
        component
        for component in ("head", "body")
        if isinstance(
            (rig := _resolve_datablock(_field(instance, f"{component}_rig"), bpy.data.objects)), bpy.types.Object
        )
        and rig.type == "ARMATURE"
        and isinstance(rig.data, bpy.types.Armature)
        and rig.data.bones
    ]


def _resolve_datablock(value: Any, collection: bpy.types.bpy_prop_collection) -> bpy.types.ID | None:
    """Resolve a rig-instance pointer field to a datablock in ``collection``.

    A live RNA pointer and a raw IDProperty ID-pointer both expose ``.name``;
    older data may instead store the datablock name as a plain string. Returns
    ``None`` when the value is empty or no matching datablock exists.
    """
    if not value:
        return None
    if isinstance(value, bpy.types.ID):
        return value
    name = value if isinstance(value, str) else getattr(value, "name", None)
    if not name:
        return None
    return collection.get(name)


def _copy_rig_instance_fields(target_properties: "CharacterSceneProperties", source: Any) -> None:  # noqa: PLR0912
    """Create a rig instance on ``target_properties`` from a migration ``source``.

    ``source`` is either a live ``RigInstance`` (sibling edition) or a raw
    IDProperty group (old prototype). Recognized fields are copied onto a freshly
    added rig instance, resolving object and material pointers by name. Handles
    both the current nested ``output`` group and the old flat ``output_folder_path``
    field.
    """
    name = _field(source, "name") or _field(source, "instance_name")
    if not name:
        return
    instance = target_properties.rig_instance_list.add()
    instance.name = name

    for field in ("head_dna_file_path", "body_dna_file_path"):
        value = _field(source, field)
        if value:
            setattr(instance, field, value)

    for field in ("face_board", "control_rig", "head_mesh", "head_rig", "body_mesh", "body_rig"):
        resolved = _resolve_datablock(_field(source, field), bpy.data.objects)
        if resolved is not None:
            setattr(instance, field, resolved)

    for field in ("head_material", "body_material"):
        resolved = _resolve_datablock(_field(source, field), bpy.data.materials)
        if resolved is not None:
            setattr(instance, field, resolved)

    # Output folder: the current format nests it under ``output``; the old
    # prototype stored it flat as ``output_folder_path``.
    output_folder_path = _field(source, "output_folder_path")
    if output_folder_path is None:
        output = _field(source, "output")
        if output is not None:
            output_folder_path = _field(output, "folder_path")
    if output_folder_path:
        instance.output.folder_path = output_folder_path

    head_to_body_constraint_influence = _field(source, "head_to_body_constraint_influence")
    if (
        head_to_body_constraint_influence is not None
        and head_to_body_constraint_influence != instance.head_to_body_constraint_influence
    ):
        instance.head_to_body_constraint_influence = head_to_body_constraint_influence

    for field in (
        "auto_evaluate",
        "auto_evaluate_head",
        "auto_evaluate_body",
        "evaluate_bones",
        "evaluate_shape_keys",
        "evaluate_texture_masks",
        "evaluate_rbfs",
    ):
        value = _field(source, field)
        if value is not None and value != getattr(instance, field):
            setattr(instance, field, value)
    identity = source.get("native_runtime_id")
    if identity:
        instance["native_runtime_id"] = identity
    view_options = _field(source, "view_options")
    if view_options is not None:
        for prop in instance.view_options.bl_rna.properties:
            if prop.identifier == "rna_type" or prop.is_readonly or prop.type in {"POINTER", "COLLECTION"}:
                continue
            value = _field(view_options, prop.identifier)
            if value is not None and value != getattr(instance.view_options, prop.identifier):
                setattr(instance.view_options, prop.identifier, value)


def migrate_by_collection_data(context: "Context", addon_id: str) -> None:
    for collection in bpy.context.collection.children_recursive:
        if collection.name.endswith("_lod0"):
            rig_instance_name = collection.name[:-5]
            if rig_instance_name not in [
                instance.name for instance in get_addon_scene_properties(context).rig_instance_list
            ]:
                instance = add_rig_instance(name=rig_instance_name)
                instance.head_rig = bpy.data.objects.get(rig_instance_name + "_head_rig")
                instance.body_rig = bpy.data.objects.get(rig_instance_name + "_body_rig")
                instance.head_mesh = bpy.data.objects.get(rig_instance_name + "_head_lod0_mesh")
                instance.body_mesh = bpy.data.objects.get(rig_instance_name + "_body_lod0_mesh")
                instance.face_board = bpy.data.objects.get(rig_instance_name + "_face_gui")
                instance.control_rig = bpy.data.objects.get(rig_instance_name + "_control_rig")
                instance.head_material = bpy.data.materials.get(rig_instance_name + "_head_shader")
                instance.body_material = bpy.data.materials.get(rig_instance_name + "_body_shader")

    # Remove old addon key in scene data after migration
    bpy.context.scene.pop(addon_id, None)


def migrate_legacy_data(
    context: "Context",
) -> Literal["default", "collection_data", "cross_edition", "legacy_format"]:
    """Convert edition metadata without changing evaluation state or rebuilding rigs."""
    from ..runtime import controller

    properties = get_addon_window_manager_properties(context)
    previous_evaluation = properties.evaluate_dependency_graph
    properties.evaluate_dependency_graph = False
    try:
        with controller.preserve_bindings():
            return _migrate_legacy_data(context)
    finally:
        properties.evaluate_dependency_graph = previous_evaluation


def _migrate_legacy_data(
    context: "Context",
) -> Literal["default", "collection_data", "cross_edition", "legacy_format"]:
    """Migrate rig-instance data saved by a different addon edition or version.

    Handles three scenarios, all keyed off :func:`detect_legacy_data`:

    * **Cross-edition** — the .blend was saved by the sibling edition
      (``character_dna`` vs ``character_dna_pro``). Both editions share the same
      ``RigInstance`` layout, so each instance is rebuilt field-by-field.
    * **Legacy format** — the old ``meta_human_dna`` prototype stored its rig
      instances under ``rig_logic_instance_list`` with the same field names but a
      flat output folder; each instance is rebuilt the same way.
    * **Collection data** — only the asset collections survived (no rig-instance
      list), so instances are reconstructed from the ``*_lod0`` collection names.

    Returns a status string describing which path ran.
    """
    scene = context.scene
    if not scene:
        return "default"

    detected = detect_legacy_data(scene)
    if not detected:
        return "default"

    addon_id, data_key = detected

    # No rig-instance list survived; rebuild from the asset collection names.
    if not data_key:
        migrate_by_collection_data(context, addon_id)
        return "collection_data"

    raw_data = get_raw_scene_data(scene, addon_id)
    sources = _rig_instance_sources(raw_data, data_key)

    target_properties = get_addon_scene_properties(context)
    existing_names = [instance.name for instance in target_properties.rig_instance_list]

    # Rebuild each instance through the RNA so the data lands in the current
    # edition's managed storage. A raw IDProperty subtree copy is not viable here:
    # registered PointerProperty data is not exposed as a plain subscriptable
    # IDProperty, so the registered property would never read it back.
    for source in sources:
        name = _field(source, "name") or _field(source, "instance_name")
        if name and name not in existing_names:
            _copy_rig_instance_fields(target_properties, source)
            existing_names.append(name)

    migrate_type: Literal["cross_edition", "legacy_format"] = (
        "cross_edition" if data_key == "rig_instance_list" else "legacy_format"
    )

    # Clear the migrated sibling list so it is neither re-detected nor re-saved.
    if addon_id != ToolInfo.NAME and hasattr(raw_data, "bl_rna"):
        raw_data.rig_instance_list.clear()

    # Remove any subscript-stored foreign/legacy keys (old prototype).
    for other_id in ADDON_IDS:
        if other_id != ToolInfo.NAME and other_id in scene:
            del scene[other_id]

    # Drop any stale legacy list key left on the current edition's group.
    current_group = getattr(scene, ToolInfo.NAME, None)
    if current_group is not None:
        for key in LEGACY_DATA_KEYS:
            if current_group.get(key) is not None:
                del current_group[key]

    return migrate_type


def get_addon_preferences() -> "CharacterAddonPreferences | None":
    """
    Gets the addon preferences for the Character DNA addon.

    Returns:
        CharacterAddonPreferences | None: The addon preferences or None if not found.
    """
    if not bpy.context.preferences:
        return None

    # use cached extension id if available
    if ToolInfo.EXTENSION_ID:
        return bpy.context.preferences.addons[ToolInfo.EXTENSION_ID].preferences  # type: ignore[attr-defined]

    # search for the addon preferences, these can be defined under different names depending on how
    # the addon was installed. E.g. "character_dna" or "bl_ext.user_default.character_dna"
    for extension_id in bpy.context.preferences.addons.keys():  # noqa: SIM118
        key = extension_id.split(".")[-1]
        if key == ToolInfo.NAME:
            ToolInfo.EXTENSION_ID = extension_id
            return bpy.context.preferences.addons[extension_id].preferences  # type: ignore[attr-defined]
    return None


def editors_available() -> bool:
    """Return ``True`` when the optional Pro ``editors`` submodule is present.

    The presence of the ``editors`` package's ``__init__.py`` is the single
    source of truth for "is this the Pro edition?". The result is cached on the
    centralized window-manager ``data`` dictionary so repeated polls are cheap.
    """
    from ..properties import CharacterWindowManagerProperties

    cache = CharacterWindowManagerProperties.data
    if "editors_available" not in cache:
        editors_init = Path(__file__).parent.parent / "editors" / "__init__.py"
        cache["editors_available"] = editors_init.is_file()
    return cache["editors_available"]


def get_editors() -> ModuleType | None:
    """Return the imported ``editors`` registry module, or ``None`` when absent.

    The resolved module (or ``None`` when the submodule is missing or fails to
    import) is cached on the centralized window-manager ``data`` dictionary so a
    failed import is not retried on every call. When the submodule is missing
    (free edition) the caller should fall back to the core-only behavior.
    """
    from ..properties import CharacterWindowManagerProperties

    cache = CharacterWindowManagerProperties.data
    if "editors_module" not in cache:
        module: ModuleType | None = None
        if editors_available():
            try:
                module = importlib.import_module("..editors", package=__package__)
            except Exception:
                logger.exception("Failed to import the editors submodule; running without it.")
        cache["editors_module"] = module
    return cache["editors_module"]


def pro_features_visible() -> bool:
    """Return ``True`` when the Pro editor UI should be shown.

    This is the case only in a Pro build (the ``editors`` submodule is present)
    *and* when the ``show_pro_features`` preview toggle is enabled. Pro users can
    disable the toggle to preview what the free edition's UI looks like.
    """
    if not editors_available():
        return False

    preferences = get_addon_preferences()
    return bool(getattr(preferences, "show_pro_features", True))


def get_addon_window_manager_properties(context: bpy.types.Context | None = None) -> "CharacterWindowManagerProperties":
    """
    Gets the window manager properties for the Character DNA addon.

    Returns:
        CharacterWindowManagerProperties: The window manager properties.
    """
    if context is None:
        context = bpy.context

    if context.window_manager and hasattr(context.window_manager, ToolInfo.NAME):
        return getattr(context.window_manager, ToolInfo.NAME)
    return None  # type: ignore[reportReturnType]


def get_addon_scene_properties(
    context: bpy.types.Context | None = None, id_override: str = ToolInfo.NAME
) -> "CharacterSceneProperties":
    """
    Gets the scene properties for the Character DNA addon.

    Returns:
        CharacterSceneProperties: The scene properties.
    """
    if context is None:
        context = bpy.context

    if context.scene and hasattr(context.scene, id_override):
        return getattr(context.scene, id_override)
    return None  # type: ignore[reportReturnType]


def get_addon_ops_module() -> ModuleType:
    """
    Gets the operator module for the Character DNA addon.

    Returns:
        ModuleType: The operator's module for the addon.
    """
    return getattr(bpy.ops, ToolInfo.NAME, None)  # type: ignore[reportReturnType]


def file_path_hash(file_path: Path, length: int = 8) -> str:
    """
    Generates a consistent hash for a file path.

    The path is normalized to ensure the same logical path always produces
    the same hash, regardless of slash direction, case (on Windows), or
    trailing separators.

    Args:
        file_path: The file path to hash.
        length: The length of the returned hash (default 8 characters).

    Returns:
        A short, consistent hash string.
    """
    # Normalize the path: resolve to absolute, normalize slashes and case
    normalized = file_path.resolve().as_posix()

    # Hash the normalized path
    byte_string = normalized.encode("utf-8")
    hex_digest = hashlib.sha256(byte_string).hexdigest()

    # Return the first N characters for a shorter hash
    return hex_digest[:length]


def disable_duplicate_addons():
    # If the pro version of the addon is enabled, disable any other versions to avoid conflicts
    if ToolInfo.NAME == "character_dna_pro":
        enabled_addons = [mod.__name__ for mod in addon_utils.modules() if addon_utils.check(mod.__name__)[1]]  # type: ignore[reportGeneralTypeIssues]
        for addon_name in enabled_addons:
            if addon_name.endswith(("meta_human_dna", "character_dna")):
                addon_utils.disable(addon_name)
