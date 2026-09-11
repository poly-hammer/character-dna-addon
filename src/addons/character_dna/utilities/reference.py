"""Object-scoped import of persisted Character DNA references."""

import hashlib

from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

import bpy


SCHEMA_VERSION = 2
MIGRATION_MESSAGE = (
    "Legacy data detected. Please open the source file, run Migrate Legacy Data, save then retry appending/linking"
)
_MARKER = "character_dna_native_carrier"
_OBJECT_FIELDS = ("head_rig", "body_rig", "head_mesh", "body_mesh", "face_board", "control_rig")
_MATERIAL_FIELDS = ("head_material", "body_material")
_FLAGS = (
    "auto_evaluate",
    "auto_evaluate_head",
    "auto_evaluate_body",
    "evaluate_bones",
    "evaluate_rbfs",
    "evaluate_shape_keys",
    "evaluate_texture_masks",
)


class LegacyDataError(ValueError):
    """Identify migration failures so the UI can report one actionable message."""


def _identity(owner: Any) -> str | None:
    if not isinstance(owner, bpy.types.ID):
        return None
    library = str(Path(bpy.path.abspath(owner.library.filepath)).resolve()) if owner.library else ""
    return f"{owner.id_type}:{library}:{owner.name}"


def _dependencies() -> dict[Any, set[Any]]:
    result: dict[Any, set[Any]] = {}
    for dependency, users in bpy.data.user_map().items():
        for user in users:
            result.setdefault(user, set()).add(dependency)
    return result


def describe_instance(scene: Any, edition: str, instance: Any) -> dict:  # noqa: PLR0912
    """Describe saved ID properties without registering or initializing the addon."""
    name = instance.get("name") or instance.get("instance_name")
    pointers = {field: instance.get(field) for field in (*_OBJECT_FIELDS, *_MATERIAL_FIELDS)}
    issues = []
    roots = [collection for collection in scene.collection.children_recursive if collection.name == name]
    root = instance.get("reference_root")
    if root is None and len(roots) == 1:
        root = roots[0]
    members = set(root.all_objects) if root else set()
    for field, owner in pointers.items():
        expected = bpy.types.Material if field in _MATERIAL_FIELDS else bpy.types.Object
        if owner is not None and not isinstance(owner, expected):
            if not hasattr(owner, "keys") or len(owner.keys()):
                issues.append(f"Invalid saved pointer: {field}")
            pointers[field] = None
    dependencies = _dependencies()
    pending = list(members | {owner for owner in pointers.values() if isinstance(owner, bpy.types.ID)})
    reachable = set()
    while pending:
        owner = pending.pop()
        if owner in reachable:
            continue
        reachable.add(owner)
        for dependency in dependencies.get(owner, ()):
            if isinstance(dependency, (bpy.types.Scene, bpy.types.Collection)):
                issues.append(f"Unsupported scene/collection dependency: {owner.name} -> {dependency.name}")
            else:
                pending.append(dependency)
    objects = {owner for owner in reachable if isinstance(owner, bpy.types.Object)}
    carriers = [owner for owner in objects if owner.get(_MARKER) == 1]
    identity = instance.get("native_runtime_id", "")
    for carrier in carriers:
        if carrier.get("schema_version") != SCHEMA_VERSION:
            issues.append(f"{carrier.name}: incompatible persisted runtime")
        if not carrier.is_property_overridable_library('["targets"]'):
            issues.append(f"{carrier.name}: native targets require migration for editable linking")
        if identity and carrier.get("instance_id") != identity:
            issues.append(f"Cross-character carrier dependency: {carrier.name}")
        if carrier.get("scene") is not None:
            issues.append(f"{carrier.name}: runtime still depends on its source scene")
        if carrier.animation_data is None or not carrier.animation_data.drivers.find('["epoch"]'):
            issues.append(f"{carrier.name}: missing saved epoch driver")
        for target in carrier.get("targets", []):
            owner = target.get("owner")
            if owner and target.get("embedded"):
                owner = owner.node_tree
            curve = (
                owner.animation_data.drivers.find(target["path"], index=max(0, target["index"]))
                if owner and owner.animation_data
                else None
            )
            if curve is None or not any(
                variable_target.id == carrier
                for variable in curve.driver.variables
                for variable_target in variable.targets
            ):
                issues.append(f"{carrier.name}: missing or disconnected output driver")
    settings = {flag: bool(instance.get(flag, True)) for flag in _FLAGS}
    settings["active_lod"] = instance.get("view_options", {}).get("active_lod", 0)
    issues.extend(
        f"Missing persisted {component} runtime"
        for component in ("head", "body")
        if pointers[f"{component}_rig"]
        and settings["auto_evaluate"]
        and settings[f"auto_evaluate_{component}"]
        and not any(carrier.get("component") == component for carrier in carriers)
    )

    layers = {}

    def collect_layers(layer: Any) -> None:
        layers[layer.collection] = layer
        for child in layer.children:
            collect_layers(child)

    collect_layers(scene.view_layers[0].layer_collection)

    def collection_data(collection: Any) -> dict:
        layer = layers.get(collection)
        return {
            "id": _identity(collection),
            "name": collection.name,
            "objects": [_identity(owner) for owner in collection.objects],
            "hide_viewport": collection.hide_viewport,
            "hide_render": collection.hide_render,
            "hide_select": collection.hide_select,
            "exclude": layer.exclude if layer else False,
            "layer_hide_viewport": layer.hide_viewport if layer else False,
            "children": [collection_data(child) for child in collection.children],
        }

    data = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "scene": scene.name,
        "edition": edition,
        "native_runtime_id": identity,
        "pointers": {field: _identity(owner) for field, owner in pointers.items()},
        "objects": [{"id": _identity(owner), "name": owner.name} for owner in sorted(objects, key=_identity)],
        "collections": collection_data(root) if root else None,
        "dependencies": [_identity(owner) for owner in objects - members],
        "materials": [
            {
                "id": _identity(owner),
                "name": owner.name,
                "slots": [
                    {"object": _identity(obj), "index": index}
                    for obj in objects
                    for index, slot in enumerate(obj.material_slots)
                    if slot.material == owner
                ],
            }
            for owner in reachable
            if isinstance(owner, bpy.types.Material)
        ],
        "carriers": [
            {
                "id": _identity(carrier),
                "schema_version": carrier.get("schema_version", 0),
                "component": carrier.get("component", ""),
            }
            for carrier in carriers
        ],
        "settings": settings,
        "issues": sorted(set(issues)),
        "output_folder_path": instance.get("output_folder_path") or instance.get("output", {}).get("folder_path", ""),
    }
    for component in ("head", "body"):
        path = instance.get(f"{component}_dna_file_path", "")
        rig = pointers[f"{component}_rig"]
        data[f"{component}_dna_file_path"] = (
            bpy.path.abspath(path, library=rig.library if rig else None) if path else ""
        )
    return data


def validate_descriptors(data: dict, names: list[str]) -> list[dict]:
    """Preflight the complete source selection before loading any datablocks."""
    selected = []
    for name in names:
        descriptor = data.get(name)
        if descriptor is None:
            raise ValueError(f"Character not found in source: {name}")
        issues = descriptor.get("issues", [])
        if descriptor.get("schema_version") != SCHEMA_VERSION or issues:
            details = "; ".join(issues) or "Missing portable runtime descriptor"
            raise LegacyDataError(f"{name}: {details}. {MIGRATION_MESSAGE}")
        if any(carrier.get("schema_version") != SCHEMA_VERSION for carrier in descriptor.get("carriers", [])):
            raise LegacyDataError(f"{name}: incompatible saved carriers. {MIGRATION_MESSAGE}")
        objects = {item["id"] for item in descriptor["objects"]}
        if descriptor["collections"] is None:
            raise ValueError(f"{name}: missing collection descriptor")
        for field in _OBJECT_FIELDS:
            pointer = descriptor["pointers"].get(field)
            if pointer is not None and pointer not in objects:
                raise ValueError(f"{name}: unresolved saved pointer {field}")
        for component in ("head", "body"):
            if descriptor["pointers"].get(f"{component}_rig"):
                path = descriptor.get(f"{component}_dna_file_path", "")
                if not path or not Path(path).is_file():
                    raise ValueError(f"{name}: missing {component} DNA file: {path}")
        selected.append(descriptor)
    return selected


def validate_names(names: Iterable[str], existing: Iterable[str]) -> list[str]:
    """Reject an invalid selection before inspecting or importing the source file."""
    selected = [name.strip() for name in names]
    if not selected or any(not name for name in selected):
        raise ValueError("Select at least one character with a non-empty name")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate character names in selection")
    conflicts = set(selected).intersection(existing)
    if conflicts:
        raise ValueError(f"Characters already exist in this scene: {', '.join(sorted(conflicts))}")
    return selected


def scene_names(scene: Any) -> set[str]:
    """Read names from every saved edition in the destination scene."""
    names = set()
    for edition in ("character_dna", "character_dna_pro", "meta_human_dna", "meta_human_dna_pro"):
        group = getattr(scene, edition, None)
        if group is None:
            group = scene.get(edition)
        if group is None:
            continue
        for key in ("rig_instance_list", "rig_logic_instance_list"):
            for instance in getattr(group, key, ()) or group.get(key, ()):
                name = instance.get("name") or instance.get("instance_name")
                if name:
                    names.add(name)
    return names


def _remap_properties(group: Any, mapping: dict, path: str = "") -> set[str]:
    changed = set()
    try:
        keys = list(group.keys())
    except TypeError as error:
        if "doesn't support IDProperties" not in str(error):
            raise
        return changed
    for key in keys:
        value = group[key]
        property_path = f'{path}["{bpy.utils.escape_identifier(key)}"]'
        if isinstance(value, bpy.types.ID):
            replacement = mapping.get(value, value)
            if replacement != value:
                group[key] = replacement
                changed.add(property_path)
        elif hasattr(value, "keys"):
            changed.update(_remap_properties(value, mapping, property_path))
        elif isinstance(value, (list, tuple)) or type(value).__name__ == "IDPropertyArray":
            for index, item in enumerate(value):
                if hasattr(item, "keys"):
                    changed.update(_remap_properties(item, mapping, f"{property_path}[{index}]"))
    return changed


def _remap_pointers(struct: Any, mapping: dict, path: str | None = None) -> set[str]:
    changed = set()
    for prop in struct.bl_rna.properties:
        if prop.type != "POINTER" or prop.is_readonly or prop.identifier == "rna_type":
            continue
        value = getattr(struct, prop.identifier)
        if isinstance(value, bpy.types.ID) and value in mapping and mapping[value] != value:
            setattr(struct, prop.identifier, mapping[value])
            changed.add(f"{path}.{prop.identifier}" if path is not None else struct.path_from_id(prop.identifier))
    return changed


def _remap_id(owner: Any, mapping: dict) -> set[str]:  # noqa: PLR0912
    parent_inverse = owner.matrix_parent_inverse.copy() if isinstance(owner, bpy.types.Object) else None
    changed = _remap_properties(owner, mapping)
    changed.update(_remap_pointers(owner, mapping))
    if parent_inverse is not None and "parent" in changed:
        owner.matrix_parent_inverse = parent_inverse
        changed.add("matrix_parent_inverse")
    animation = getattr(owner, "animation_data", None)
    if animation:
        for curve_index, curve in enumerate(animation.drivers):
            for variable_index, variable in enumerate(curve.driver.variables):
                for target_index, target in enumerate(variable.targets):
                    path = (
                        f"animation_data.drivers[{curve_index}].driver.variables[{variable_index}]"
                        f".targets[{target_index}]"
                    )
                    changed.update(_remap_pointers(target, mapping, path))
    if isinstance(owner, bpy.types.Object):
        for struct in (*owner.constraints, *owner.modifiers):
            changed.update(_remap_pointers(struct, mapping))
            if hasattr(struct, "keys"):
                changed.update(_remap_properties(struct, mapping, struct.path_from_id()))
            for target in getattr(struct, "targets", ()):
                changed.update(_remap_pointers(target, mapping))
        if owner.pose:
            for bone in owner.pose.bones:
                changed.update(_remap_properties(bone, mapping, bone.path_from_id()))
                changed.update(_remap_pointers(bone, mapping))
                for constraint in bone.constraints:
                    changed.update(_remap_pointers(constraint, mapping))
                    for target in getattr(constraint, "targets", ()):
                        changed.update(_remap_pointers(target, mapping))
        for slot in owner.material_slots:
            if slot.material in mapping:
                slot.material = mapping[slot.material]
                changed.add(slot.path_from_id("material"))
    for field in ("node_tree", "shape_keys"):
        embedded = getattr(owner, field, None)
        if embedded and (embedded.is_embedded_data or isinstance(embedded, bpy.types.Key)):
            changed.update(f"{field}.{path}" for path in _remap_id(embedded, mapping))
    if isinstance(owner, bpy.types.NodeTree):
        for node in owner.nodes:
            changed.update(_remap_pointers(node, mapping))
            for socket in node.inputs:
                changed.update(_remap_pointers(socket, mapping))
    materials = getattr(owner, "materials", None)
    if materials is not None:
        for index, material in enumerate(materials):
            if material in mapping:
                materials[index] = mapping[material]
                changed.add(f"materials[{index}]")
    return changed


def _persist_override(owner: Any, paths: Iterable[str]) -> None:
    """Keep scoped remaps when Blender rebuilds a protected ID from its reference."""
    override = owner.override_library
    if override is None:
        return
    for path in sorted(paths):
        override.properties.add(path).operations.add("REPLACE")
    override.operations_update()


def _copy_animation(owner: Any, mapping: dict) -> None:
    animation = getattr(owner, "animation_data", None)
    if not animation:
        return
    for assignment in [animation, *(strip for track in animation.nla_tracks for strip in track.strips)]:
        action = assignment.action
        if action is None:
            continue
        slot_handle = getattr(assignment, "action_slot_handle", None)
        if action not in mapping:
            mapping[action] = action.copy()
        assignment.action = mapping[action]
        if slot_handle is not None:
            assignment.action_slot_handle = slot_handle


def _editable_objects(objects: dict[str, Any], descriptor: dict) -> dict[str, Any]:  # noqa: PLR0912
    for owner in objects.values():
        if owner.get(_MARKER) == 1 and not owner.is_property_overridable_library('["targets"]'):
            raise LegacyDataError(f"{owner.name}: saved native targets are not overridable. {MIGRATION_MESSAGE}")
    dependencies = _dependencies()
    reachable = set()
    pending = list(objects.values())
    while pending:
        owner = pending.pop()
        if owner in reachable or isinstance(owner, (bpy.types.Library, bpy.types.Scene, bpy.types.Collection)):
            continue
        reachable.add(owner)
        pending.extend(dependencies.get(owner, ()))
    mapping = {}
    for field in ("face_board", "control_rig"):
        source = objects.get(descriptor["pointers"].get(field))
        if source is None or source in mapping:
            continue
        copied = source.copy()
        mapping[source] = copied
        if source.data:
            if source.data not in mapping:
                mapping[source.data] = source.data.copy()
            copied.data = mapping[source.data]
            _copy_animation(copied.data, mapping)
        _copy_animation(copied, mapping)
    changed = set(mapping) | {owner for owner in objects.values() if owner.get(_MARKER) == 1}
    while True:
        added = {owner for owner in reachable - changed if dependencies.get(owner, set()).intersection(changed)}
        if not added:
            break
        changed.update(added)
    priority = {"OBJECT": 0, "MESH": 1, "ARMATURE": 1, "CURVES": 1, "MATERIAL": 2, "NODETREE": 3}
    for owner in sorted(changed, key=lambda item: (priority.get(item.id_type, 4), item.name)):
        if owner in mapping or owner.is_embedded_data or isinstance(owner, (bpy.types.Action, bpy.types.Key)):
            continue
        if owner.library:
            override = owner.override_create(remap_local_usages=False)
            if override is None:
                raise ValueError(f"Cannot create required protected override: {owner.id_type} {owner.name}")
            override.override_library.is_system_override = True
            mapping[owner] = override
    for source, destination in list(mapping.items()):
        for field in ("node_tree", "shape_keys"):
            old = getattr(source, field, None)
            new = getattr(destination, field, None)
            if old and new and old != new and old not in mapping:
                mapping[old] = new
    for destination in set(mapping.values()):
        if destination.is_embedded_data or isinstance(destination, bpy.types.Key):
            continue
        _persist_override(destination, _remap_id(destination, mapping))
    return {identity: mapping.get(owner, owner) for identity, owner in objects.items()}


def _local_collections(descriptor: dict, objects: dict[str, Any]) -> Any:
    collections = {}
    widgets = _widget_objects(objects.get(descriptor["pointers"].get("face_board")), set(objects.values()))

    def create(item: dict) -> Any:
        if item["id"] in collections:
            return collections[item["id"]]
        collection = bpy.data.collections.new(item["name"])
        collections[item["id"]] = collection
        for flag in ("hide_viewport", "hide_render", "hide_select"):
            setattr(collection, flag, item[flag])
        for identity in item["objects"]:
            if objects[identity] not in widgets:
                collection.objects.link(objects[identity])
        for child in item["children"]:
            collection.children.link(create(child))
        return collection

    root = create(descriptor["collections"])
    dependencies = {objects[identity] for identity in descriptor["dependencies"]} - widgets
    if dependencies:
        helpers = bpy.data.collections.new(f"{descriptor['name']}_reference_dependencies")
        root.children.link(helpers)
        for owner in dependencies:
            helpers.objects.link(owner)
    return root


def _widget_objects(face: Any, objects: set) -> set:
    if face is None or face.pose is None:
        return set()
    widgets = {bone.custom_shape for bone in face.pose.bones if bone.custom_shape}
    while True:
        parents = {
            owner.parent
            for owner in widgets
            if owner.parent in objects
            and owner.parent != face
            and owner.parent.type == "EMPTY"
            and all(child in widgets for child in owner.parent.children)
        } - widgets
        if not parents:
            return widgets
        widgets.update(parents)


def _widget_signature(shape: Any) -> tuple | None:
    if shape.type == "EMPTY":
        return ("EMPTY", shape.empty_display_type, shape.empty_display_size)
    if shape.type != "MESH" or shape.modifiers or shape.data.shape_keys:
        return None
    mesh = shape.data
    digest = hashlib.sha256()
    for values, property_name, width, code in (
        (mesh.vertices, "co", 3, "f"),
        (mesh.edges, "vertices", 2, "i"),
        (mesh.loops, "vertex_index", 1, "i"),
        (mesh.polygons, "loop_total", 1, "i"),
    ):
        buffer = array(code, [0]) * (len(values) * width)
        values.foreach_get(property_name, buffer)
        digest.update(buffer.tobytes())
    return ("MESH", digest.digest())


def _share_face_widgets(faces: list, previous_faces: list, imported: set) -> None:
    canonical = {}
    signatures = {}

    def key(bone: Any) -> tuple | None:
        shape = bone.custom_shape
        if shape not in signatures:
            signatures[shape] = _widget_signature(shape)
        signature = signatures[shape]
        return (bone.name, signature) if signature is not None else None

    for face in previous_faces:
        for bone in face.pose.bones:
            if bone.custom_shape and (identity := key(bone)) is not None:
                canonical.setdefault(identity, bone.custom_shape)
    replacements = {}
    for face in faces:
        for bone in face.pose.bones:
            shape = bone.custom_shape
            if not shape or (identity := key(bone)) is None:
                continue
            existing = canonical.setdefault(identity, shape)
            if existing != shape and not face.library:
                bone.custom_shape = existing
                _persist_override(face, (bone.path_from_id("custom_shape"),))
                replacements[shape] = existing
    removable = {
        shape for shape in replacements if shape in imported and not shape.library and not shape.override_library
    }
    if removable:
        for owner in imported - removable:
            if not owner.library and owner.parent in replacements:
                inverse = owner.matrix_parent_inverse.copy()
                owner.parent = replacements[owner.parent]
                owner.matrix_parent_inverse = inverse
        unused_meshes = {shape.data for shape in removable if shape.data}
        bpy.data.batch_remove(ids=list(removable))
        bpy.data.batch_remove(ids=[mesh for mesh in unused_meshes if mesh.users == 0])


def _restore_layer_visibility(view_layer: Any, root: Any, descriptor: dict) -> None:
    def apply(layer: Any, item: dict) -> None:
        layer.hide_viewport = item.get("layer_hide_viewport", False)
        for child, child_item in zip(layer.children, item["children"], strict=False):
            apply(child, child_item)
        layer.exclude = item.get("exclude", False)

    def find(layer: Any) -> None:
        if layer.collection == root:
            apply(layer, descriptor["collections"])
            return
        for child in layer.children:
            find(child)

    find(view_layer.layer_collection)


def _load_objects(file_path: str, descriptors: list[dict], link: bool, relative: bool) -> dict:
    entries = {item["id"]: item["name"] for descriptor in descriptors for item in descriptor["objects"]}
    if len(set(entries.values())) != len(entries):
        raise ValueError("Ambiguous source object names across libraries")
    with bpy.data.libraries.load(file_path, link=link, relative=relative) as (data_from, data_to):
        missing = set(entries.values()) - set(data_from.objects)
        if missing:
            raise ValueError(f"Missing source objects: {', '.join(sorted(missing))}")
        data_to.objects = list(entries.values())
    if any(owner is None for owner in data_to.objects):
        raise ValueError("Source object import was incomplete")
    return dict(zip(entries, data_to.objects, strict=True))


def import_characters(  # noqa: PLR0912, PLR0915
    context: Any, file_path: str, descriptors: list[dict], mode: str, relative: bool
) -> list:
    """Stage object-only references, adopt saved drivers, and roll back only new IDs."""
    from .. import utilities
    from ..runtime import controller, engine

    for api in ("binding_issues", "adopt", "release_records"):
        if not callable(getattr(engine, api, None)):
            raise RuntimeError(f"Portable reference runtime API is unavailable: engine.{api}")
    before = set(bpy.data.user_map())
    properties = utilities.get_addon_scene_properties(context)
    previous_index = properties.rig_instance_list_active_index
    previous_faces = [instance.face_board for instance in properties.rig_instance_list if instance.face_board]
    instances = []
    try:
        loaded = _load_objects(file_path, descriptors, mode != "APPEND", relative)
        unexpected = [
            owner
            for owner in set(bpy.data.user_map()) - before
            if isinstance(owner, (bpy.types.Scene, bpy.types.Collection))
        ]
        if unexpected:
            raise ValueError("Source objects pulled in a scene or collection dependency")
        for descriptor in descriptors:
            objects = {item["id"]: loaded[item["id"]] for item in descriptor["objects"]}
            if mode == "EDITABLE_LINK":
                objects = _editable_objects(objects, descriptor)
            root = _local_collections(descriptor, objects)
            instance = properties.rig_instance_list.add()
            instances.append(instance)
            instance["name"] = descriptor["name"]
            instance["old_name"] = descriptor["name"]
            instance["reference_root"] = root
            instance["reference_mode"] = mode
            instance["reference_source"] = file_path
            identity = uuid4().hex if mode != "LINK" else descriptor["native_runtime_id"]
            instance["native_runtime_id"] = identity
            for field in _OBJECT_FIELDS:
                owner = objects.get(descriptor["pointers"].get(field))
                if owner is not None:
                    instance[field] = owner
            for component in ("head", "body"):
                instance[f"{component}_dna_file_path"] = descriptor[f"{component}_dna_file_path"]
                source_material = descriptor["pointers"].get(f"{component}_material")
                if source_material:
                    material = next(item for item in descriptor["materials"] if item["id"] == source_material)
                    matches = {
                        objects[slot["object"]].material_slots[slot["index"]].material for slot in material["slots"]
                    }
                    if len(matches) != 1:
                        raise ValueError(f"Cannot resolve imported {component} material: {material['name']}")
                    instance[f"{component}_material"] = matches.pop()
            instance.output["folder_path"] = descriptor["output_folder_path"]
            for flag in _FLAGS:
                instance[flag] = descriptor["settings"][flag]
            instance.view_options["active_lod"] = descriptor["settings"]["active_lod"]
            if mode != "LINK":
                for item in descriptor["carriers"]:
                    carrier = objects[item["id"]]
                    carrier["instance_id"] = identity
                    carrier["token"] = uuid4().hex
                    _persist_override(carrier, ('["instance_id"]', '["token"]'))
            context.scene.collection.children.link(root)
            issues = engine.binding_issues(instance)
            if issues:
                raise ValueError("; ".join(issues))
            count = engine.adopt(instance)
            if count != len(descriptor["carriers"]):
                raise ValueError(f"Adopted {count} of {len(descriptor['carriers'])} saved native carriers")
            if mode == "APPEND":
                with controller.preserve_bindings():
                    instance.initialize()
            _restore_layer_visibility(context.view_layer, root, descriptor)
        context.view_layer.update()
    except Exception:
        for instance in instances:
            engine.release_records(instance)
        for _instance in reversed(instances):
            properties.rig_instance_list.remove(len(properties.rig_instance_list) - 1)
        properties.rig_instance_list_active_index = previous_index
        bpy.data.batch_remove(ids=set(bpy.data.user_map()) - before)
        raise
    imported_objects = {owner for owner in set(bpy.data.user_map()) - before if isinstance(owner, bpy.types.Object)}
    _share_face_widgets(
        [instance.face_board for instance in instances if instance.face_board], previous_faces, imported_objects
    )
    properties.rig_instance_list_active_index = len(properties.rig_instance_list) - 1
    for instance in instances:
        utilities.notify_rig_instances_changed(instance)
    return instances
