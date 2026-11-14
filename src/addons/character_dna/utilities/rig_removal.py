"""Bake LOD meshes before removing the character's rig and collection hierarchy."""

import ast
import re

from contextlib import contextmanager
from typing import Any

import bpy


def _drivers(owner: Any) -> list:
    animation = owner.animation_data if owner else None
    return list(animation.drivers) if animation else []


def _freeze_drivers(source: Any, target: Any, depsgraph: bpy.types.Depsgraph) -> None:
    evaluated = source.evaluated_get(depsgraph)
    for curve in _drivers(source):
        path = curve.data_path
        value = evaluated.path_resolve(path)
        if path.endswith("]"):
            container, key = path.rsplit("[", 1)
            parent = target.path_resolve(container) if container else target
            key = ast.literal_eval(key[:-1])
            current = parent[key]
        else:
            container, _, key = path.rpartition(".")
            parent = target.path_resolve(container) if container else target
            current = getattr(parent, key)
        if hasattr(value, "__len__") and not isinstance(value, str):
            current[curve.array_index] = value[curve.array_index]
        elif path.endswith("]"):
            parent[key] = value
        else:
            setattr(parent, key, value)
    target.animation_data_clear()


def _driven_tree(tree: Any, seen: set) -> bool:
    if tree is None or tree in seen:
        return False
    seen.add(tree)
    return bool(_drivers(tree)) or any(
        _driven_tree(node.node_tree, seen) for node in tree.nodes if node.type == "GROUP"
    )


def _static_material(material: Any, depsgraph: bpy.types.Depsgraph, copies: dict) -> Any:
    if material is None:
        return None
    if material in copies:
        return copies[material]
    if not _drivers(material) and not _driven_tree(material.node_tree, set()):
        return material
    result = copies[material] = material.copy()
    _freeze_drivers(material, result, depsgraph)

    def freeze_tree(source: Any, target: Any) -> None:
        _freeze_drivers(source, target, depsgraph)
        for node in target.nodes:
            if node.type != "GROUP" or not _driven_tree(node.node_tree, set()):
                continue
            group = node.node_tree
            if group not in copies:
                copies[group] = group.copy()
                freeze_tree(group, copies[group])
            node.node_tree = copies[group]

    if material.node_tree:
        freeze_tree(material.node_tree, result.node_tree)
    return result


@contextmanager
def _evaluation_visibility(context: bpy.types.Context, collections: list, objects: set):
    """Evaluate disabled LODs too, restoring display state after capture."""
    states = []

    def expose(layer: Any) -> bool:
        relevant = layer.collection in collections
        for child in layer.children:
            relevant = expose(child) or relevant
        if relevant:
            states.append((layer, "exclude", layer.exclude))
            layer.exclude = False
        return relevant

    expose(context.view_layer.layer_collection)
    for owner in [*collections, *objects]:
        states.append((owner, "hide_viewport", owner.hide_viewport))
        owner.hide_viewport = False
    try:
        context.view_layer.update()
        yield
    finally:
        for owner, field, value in reversed(states):
            setattr(owner, field, value)
        context.view_layer.update()


def _owned_ids(objects: set) -> set:
    ids = set(objects)

    def collect_tree(tree: Any) -> None:
        if tree is None or tree in ids:
            return
        ids.add(tree)
        for node in tree.nodes:
            if node.type == "GROUP":
                collect_tree(node.node_tree)

    for obj in objects:
        if obj.data:
            ids.add(obj.data)
        if obj.type == "MESH":
            ids.update(slot.material for slot in obj.material_slots if slot.material)
            if obj.data.shape_keys:
                ids.add(obj.data.shape_keys)
        baseline = obj.get("dna_merge_baseline")
        if isinstance(baseline, bpy.types.ID):
            ids.add(baseline)
    for owner in list(ids):
        if isinstance(owner, bpy.types.Material):
            collect_tree(owner.node_tree)
    for owner in list(ids):
        if owner.animation_data and owner.animation_data.action:
            ids.add(owner.animation_data.action)
    # Blender deletes shape keys and embedded node trees with their owner.
    return {owner for owner in ids if not isinstance(owner, bpy.types.Key) and not owner.is_embedded_data}


def _capture_meshes(context: bpy.types.Context, collections: list, owned: set, meshes: set) -> dict:
    captured, copies, disabled = {}, {}, []
    try:
        with _evaluation_visibility(context, collections, owned):
            # Bake the stack through its last armature. Modifiers after skinning
            # remain editable, and shape keys become their current mixed shape.
            for obj in meshes:
                count = max((i + 1 for i, mod in enumerate(obj.modifiers) if mod.type == "ARMATURE"), default=0)
                for mod in list(obj.modifiers)[count:]:
                    disabled.append((mod, mod.show_viewport))
                    mod.show_viewport = False
                captured[obj] = [None, None, count]
            context.view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            for obj, data in captured.items():
                evaluated = obj.evaluated_get(depsgraph)
                data[0] = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
                data[1] = evaluated.matrix_world.copy()
                data[0].materials.clear()
                for slot in obj.material_slots:
                    data[0].materials.append(_static_material(slot.material, depsgraph, copies))
    except Exception:
        bpy.data.batch_remove(ids={data[0] for data in captured.values() if data[0]} | set(copies.values()))
        raise
    finally:
        for mod, enabled in disabled:
            mod.show_viewport = enabled

    return captured


def bake_lod_meshes(context: bpy.types.Context, instance: Any, collections: list, objects: set) -> tuple[set, set]:
    """Prepare all geometry before committing; return kept objects and scoped orphan candidates."""
    from ..runtime import engine

    if any(collection.library or collection.override_library for collection in collections):
        raise ValueError("Append the character from its source file before applying its mesh shape")
    if any(
        collection in scene.collection.children_recursive
        for scene in bpy.data.scenes
        if scene != context.scene
        for collection in collections
    ):
        raise ValueError("The character collections are shared with another scene; make a separate copy first")
    owned = objects | {obj for collection in collections for obj in collection.objects} | set(engine.carriers(instance))
    lods = [collection for collection in collections if re.search(r"_lod\d+(?:\.\d+)?$", collection.name)]
    meshes = {obj for collection in lods for obj in collection.all_objects if obj.type == "MESH"}
    if not meshes:
        raise ValueError("No mesh LOD collections were found for this rig instance")
    if any(obj.library or obj.override_library or obj.mode != "OBJECT" for obj in meshes):
        raise ValueError("Apply mesh shape requires local meshes in Object Mode")
    visibility = {obj: obj.visible_get(view_layer=context.view_layer) for obj in meshes}
    candidates = _owned_ids(owned)
    captured = _capture_meshes(context, collections, owned, meshes)

    kept = set()
    for obj, (mesh, matrix, count) in captured.items():
        # Shared objects outside this asset retain their rigged original.
        result = obj.copy() if any(collection not in collections for collection in obj.users_collection) else obj
        result.data = mesh
        for slot in result.material_slots:
            slot.link = "DATA"
        for mod in list(result.modifiers)[:count]:
            result.modifiers.remove(mod)
        result.animation_data_clear()
        result.constraints.clear()
        result.parent = None
        result.matrix_world = matrix
        mesh.animation_data_clear()
        for name in list(result.keys()):
            if name.startswith(("dna_merge_", "dna_shape_", "ca_dna_")) or name == "dna_working_source":
                del result[name]
        if result.name not in context.scene.collection.objects:
            context.scene.collection.objects.link(result)
        result.hide_set(not visibility[obj])
        kept.add(result)
    objects.update(owned)
    return kept, candidates


def remove_orphans(objects: set, candidates: set) -> None:
    """Remove asset-local dependency cycles, then only unused owned datablocks."""
    doomed = {obj for obj in objects if obj.library is None and not obj.users_collection}
    candidates.difference_update(doomed)
    bpy.data.batch_remove(ids=doomed)
    while unused := {owner for owner in candidates if owner.library is None and owner.users == 0}:
        candidates.difference_update(unused)
        bpy.data.batch_remove(ids=unused)
