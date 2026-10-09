"""Legacy metadata, DNA coordinates, animation, and control-rig migration.

Migration stages replacement armatures and actions before rebuilding runtime
drivers. Optional control-rig rebuilding uses the installed builder's public API
and verifies animation before replacing the original controls.

The old importer changed the armature's object axes, but left Maya bone-local
axes intact. Re-reading DNA supplies the new rest frames. Animation deltas must
change basis too: B_new = N^-1 O B_old O^-1 N. Meshes and object transforms are
already in Blender space and must not be rotated a second time.
"""

from __future__ import annotations

import importlib
import math
import sys
import uuid

from contextlib import contextmanager
from dataclasses import dataclass, field as data_field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import bpy

from mathutils import Euler, Matrix, Quaternion

from ..constants import ADDON_IDS, LEGACY_DATA_KEYS, MIGRATABLE_DATA_KEYS, SCALE_FACTOR, ComponentType, ToolInfo
from .misc import add_rig_instance, get_addon_scene_properties, get_addon_window_manager_properties


if TYPE_CHECKING:
    from ..typing import CharacterSceneProperties, Context


BASIS = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
CONTROL_RIG_MESSAGE = "You must remove the control rig, then run legacy migration and rebuild the control rig."
# Compare motion in meaningful units: 0.1 mm, 0.1 degree, and 0.01% scale.
# A matrix coefficient mixes translation and rotation and exaggerates tiny
# finger-axis changes introduced by re-reading quantized DNA rest transforms.
_CONTROL_MOTION_LIMITS = (0.0001, math.radians(0.1), 0.0001)


# Legacy metadata and rig discovery


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


def field(source: Any, name: str) -> Any:
    value = _field(source, name)
    return _resolve_datablock(value, bpy.data.objects) if name.endswith(("_rig", "_mesh")) else value


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


def legacy_rigs(source: Any) -> list[tuple[ComponentType, Any]]:
    from ..dna_io.coordinates import COORDINATE_VERSION

    return [
        (component, rig)
        for component in ("head", "body")
        if (rig := field(source, f"{component}_rig"))
        and rig.type == "ARMATURE"
        and rig.data.bones
        and rig.get("dna_coordinate_version", 0) != COORDINATE_VERSION
    ]


def needs_coordinates(source: Any) -> bool:
    return bool(legacy_rigs(source))


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
        if _runtime_migration_components(instance) and (needs_coordinates(instance) or engine.binding_issues(instance)):
            return True
    return False


def runtime_migration_sources(context: Context) -> list[Any]:
    """Inspect current and unconverted records without consuming legacy metadata."""
    from ..runtime import engine

    properties = get_addon_scene_properties(context)
    instances = list(properties.rig_instance_list)
    targets = [
        instance
        for instance in instances
        if _runtime_migration_components(instance) and (needs_coordinates(instance) or engine.binding_issues(instance))
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
    return [source for source in sources if _runtime_migration_components(source)]


def _copy_rig_instance_fields(target_properties: CharacterSceneProperties, source: Any) -> None:  # noqa: PLR0912
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


def migrate_by_collection_data(context: Context, addon_id: str) -> None:
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
    context: Context,
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
    context: Context,
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


# Migration transaction and validation


def preflight_runtime_migration(instances: list[Any], *, rebuild_controls: bool = False) -> None:  # noqa: PLR0912
    """Validate a batch before creating backups or mutating rigs and metadata."""
    from ..dna_io.coordinates import COORDINATE_VERSION
    from ..runtime import bindings, engine
    from .armature import get_body_constraint_name

    issues = []
    for instance in instances:
        name = _field(instance, "name") or _field(instance, "instance_name")
        if (
            needs_coordinates(instance)
            and field(instance, "control_rig")
            and (not rebuild_controls or not control_rebuild_available(bpy.context, instance))
        ):
            issues.append(f"{name}: {CONTROL_RIG_MESSAGE}")
        carriers = engine.carriers(instance)
        for carrier in carriers:
            version = carrier.get("schema_version", 1)
            if not isinstance(version, int) or version > engine.SCHEMA_VERSION:
                issues.append(f"{name}: unsupported future runtime schema {version}; update Character DNA")
        owners = list(carriers)
        scene = getattr(instance, "id_data", None)
        if scene is not None:
            owners.append(scene)
        for field_name in ("head_rig", "body_rig", "face_board", "control_rig", "head_mesh", "body_mesh"):
            owner = _resolve_datablock(_field(instance, field_name), bpy.data.objects)
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
            rig = _resolve_datablock(_field(instance, f"{component}_rig"), bpy.data.objects)
            assert isinstance(rig, bpy.types.Object)
            version = rig.get("dna_coordinate_version", 0)
            if not isinstance(version, int) or version > COORDINATE_VERSION:
                issues.append(f"{name}: unsupported coordinate version {version}; update Character DNA")
            if rig.animation_data and rig.animation_data.use_tweak_mode:
                issues.append(f"{name}: exit NLA Tweak Mode before migrating")
            if version == 0 and rig.animation_data:
                owned = {
                    target["path"]
                    for carrier in carriers
                    for target in carrier.get("targets", ())
                    if target.get("owner") == rig
                }
                if any(
                    curve.data_path.startswith("pose.bones[") and curve.data_path not in owned
                    for curve in rig.animation_data.drivers
                ):
                    issues.append(f"{rig.name}: bake custom bone drivers to an Action before migrating their axes")
            if version == 0:
                for bone in rig.pose.bones:
                    if any(
                        constraint.name != get_body_constraint_name(bone.name)
                        and not (
                            rebuild_controls and field(instance, "control_rig") and constraint.name.startswith("ccr")
                        )
                        for constraint in bone.constraints
                    ):
                        issues.append(f"{rig.name}: bake custom bone constraints to Actions before migrating")
                        break
            path = _field(instance, f"{component}_dna_file_path")
            if not path or not Path(bpy.path.abspath(path)).is_file():
                issues.append(
                    f"{name}: missing {component} DNA file path ({path or 'not set'}); assign an existing .dna"
                )
    if issues:
        raise ValueError("Cannot migrate rig runtime: " + "; ".join(issues))


def migrate_runtime_data(  # noqa: PLR0912
    context: Context, *, migrate_actions: bool = True, rebuild_controls: bool = False
) -> tuple[str, int, int]:
    """Preflight and stage the whole batch before changing the artist's rigs."""
    from ..runtime import controller, engine

    properties = get_addon_scene_properties(context)
    editing = [instance.name for instance in properties.rig_instance_list if controller.is_suspended(instance)]
    if editing:
        raise ValueError(f"Commit or revert active edits before migrating: {', '.join(editing)}")
    eligible = runtime_migration_sources(context)
    names = {_field(source, "name") or _field(source, "instance_name") for source in eligible}
    preflight_runtime_migration(eligible, rebuild_controls=rebuild_controls)
    if eligible:
        available, reason = engine.capability()
        if not available:
            raise RuntimeError(reason)
    plans, controls, targets = [], [], []
    success = False
    with preserved_scene(context):
        try:
            for source in eligible:
                plans.extend(prepare(source, migrate_actions))
                if needs_coordinates(source) and field(source, "control_rig"):
                    controls.append(ControlUpgrade.prepare(context, source, migrate_actions))
            migrate_type = migrate_legacy_data(context)
            targets = [instance for instance in properties.rig_instance_list if instance.name in names]
            for control in controls:
                body_data = next((plan.new_data for plan in plans if plan.rig == control.body), control.body.data)
                control.stage(body_data)
            for plan in plans:
                plan.apply(migrate_actions)
            for control in controls:
                control.apply()
            for instance in targets:
                controller.rebuild(instance)
                issues = engine.binding_issues(instance)
                if issues:
                    raise RuntimeError(f"{instance.name}: runtime migration incomplete: {'; '.join(issues)}")
            for plan in plans:
                plan.restore_pose()
            for control in controls:
                control.verify()
            success = True
            return migrate_type, len(targets), len(properties.rig_instance_list) - len(targets)
        finally:
            if not success:
                affected = {plan.rig for plan in plans if plan.applied}
                for instance in targets:
                    if instance.head_rig in affected or instance.body_rig in affected:
                        # A verification failure can happen after installing the new
                        # runtime. Do not leave it writing native values to old axes.
                        engine.discard(instance)
                        instance.destroy_head()
                        instance.destroy_body()
                for control in reversed(controls):
                    control.rollback()
                for plan in reversed(plans):
                    plan.rollback()
            for control in controls:
                control.close(success)
            for plan in plans:
                plan.close(success)


def save_migration_backup() -> Path:
    """Write a uniquely named copy before the user-confirmed control rebuild."""
    original = Path(bpy.data.filepath) if bpy.data.filepath else Path(bpy.app.tempdir) / "Unsaved.blend"
    path = original.with_name(f"{original.stem}.pre-migration-{uuid.uuid4().hex[:8]}.blend")
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True, check_existing=False)
    if "FINISHED" not in result or not path.is_file():
        raise OSError("Could not save the pre-migration backup; no rigs were replaced")
    return path


@contextmanager
def preserved_scene(context: Any):
    """Preserve the artist's context without restoring obsolete Action pointers."""
    from ..runtime import controller

    scene = context.scene
    frame, subframe = scene.frame_current, scene.frame_subframe
    active, selected = context.view_layer.objects.active, list(context.selected_objects)
    active_name = active.name if active else None
    selected_names = [obj.name for obj in selected]
    mode = active.mode if active else "OBJECT"
    wm = get_addon_window_manager_properties(context)
    evaluation = wm.evaluate_dependency_graph
    wm.evaluate_dependency_graph = False
    with controller.preserve_bindings():
        try:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            yield
        finally:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            for obj in context.selected_objects:
                obj.select_set(False)
            for name in selected_names:
                obj = context.view_layer.objects.get(name)
                if obj:
                    obj.select_set(True)
            active = context.view_layer.objects.get(active_name) if active_name else None
            if active:
                context.view_layer.objects.active = active
                if mode != "OBJECT":
                    bpy.ops.object.mode_set(mode=mode)
            scene.frame_set(frame, subframe=subframe)
            wm.evaluate_dependency_graph = evaluation


# DNA armatures and animation


def slot_handle(assignment: Any) -> int:
    slot = getattr(assignment, "action_slot", None)
    return slot.handle if slot else 0


def channel_containers(action: Any, handle: int) -> list[Any]:
    if action.is_action_layered:
        return [
            bag
            for layer in action.layers
            for strip in layer.strips
            for bag in strip.channelbags
            if bag.slot_handle == handle
        ]
    return [action]


def assignments(rig: Any) -> list[Any]:
    animation = rig.animation_data
    if not animation:
        return []

    def strips(items: Any):
        for strip in items:
            if strip.type == "META":
                yield from strips(strip.strips)
            elif strip.action:
                yield strip

    return ([animation] if animation.action else []) + [
        strip for track in animation.nla_tracks for strip in strips(track.strips)
    ]


def assigned_actions(rig: Any) -> list[tuple[Any, int]]:
    return list(dict.fromkeys((item.action, slot_handle(item)) for item in assignments(rig)))


def action_labels(sources: list[Any]) -> list[str]:
    result = []
    for source in sources:
        rigs = [rig for _, rig in legacy_rigs(source)]
        control = field(source, "control_rig")
        if rigs and control:
            rigs.append(control)
        for rig in rigs:
            result.extend(f"{rig.name}: {action.name}" for action, _ in assigned_actions(rig))
    return list(dict.fromkeys(result))


def set_action(assignment: Any, action: Any, handle: int = 0) -> None:
    """Keep the assigned slot and NLA timing; assigning the same slot crashes 5.2."""
    timing = None
    if isinstance(assignment, bpy.types.NlaStrip):
        timing = {
            name: getattr(assignment, name)
            for name in ("action_frame_start", "action_frame_end", "scale", "repeat", "frame_start", "frame_end")
        }
    assignment.action = action
    if action and action.is_action_layered:
        slot = next((slot for slot in action.slots if slot.handle == handle), None)
        if slot is None:
            raise ValueError(f"Missing animation slot {handle} on {action.name}")
        if assignment.action_slot != slot:
            assignment.action_slot = slot
    if timing:
        for name, value in timing.items():
            setattr(assignment, name, value)


def sample_frames(action: Any, handle: int, owner: Any = None) -> list[float]:
    frames = {
        float(point.co.x)
        for bag in channel_containers(action, handle)
        for curve in bag.fcurves
        for point in curve.keyframe_points
    }
    start, end = action.frame_range
    frames.update((float(start), float(end)))
    frames.update(range(math.floor(start), math.ceil(end) + 1))
    if owner:
        # A stretched/repeated NLA clip samples fractional Action frames even on
        # integer scene frames. Include those times in any required rebake.
        for strip in assignments(owner):
            if not isinstance(strip, bpy.types.NlaStrip) or strip.action != action or slot_handle(strip) != handle:
                continue
            length = strip.action_frame_end - strip.action_frame_start
            if length <= 0 or strip.scale <= 0 or strip.use_animated_time:
                continue
            for frame in range(math.ceil(strip.frame_start), math.floor(strip.frame_end) + 1):
                offset = ((frame - strip.frame_start) / strip.scale) % length
                frames.add(strip.action_frame_end - offset if strip.use_reverse else strip.action_frame_start + offset)
    return sorted(frames)


def _activate(rig: Any) -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.selected_objects or ():
        obj.select_set(False)
    rig.hide_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig


def _reader(source: Any, component: str) -> Any:
    from ..dna_io.misc import get_dna_reader

    path = Path(bpy.path.abspath(field(source, f"{component}_dna_file_path")))
    reader = get_dna_reader(path, file_format="json" if path.suffix.lower() == ".json" else "binary")
    if reader is None or reader.getJointCount() == 0:
        raise ValueError(f"{path}: DNA contains no readable skeleton")
    return reader


def _stage_armature(source: Any, component: ComponentType, rig: Any, reader: Any) -> Any:
    """Import into scratch data, then transplant DNA rest frames into a copy.

    Copying the existing armature retains bone collections, custom bones,
    deform settings and properties; the existing Object retains all mesh users.
    """
    from ..bindings import enums
    from ..dna_io.importer import DNAImporter

    data = bpy.data.armatures.new(".DNA migration import")
    temporary = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(temporary)
    replacement = None
    succeeded = False
    try:
        unit = enums.TranslationUnit(reader.getTranslationUnit()).name.lower()  # pyright: ignore[reportCallIssue]
        importer_instance: Any = SimpleNamespace(name=rig.name)
        importer = DNAImporter(
            importer_instance,
            get_addon_window_manager_properties(),
            1 / SCALE_FACTOR if unit == "cm" else 1,
            component_type=component,
            create_extra_bones=False,
            reader=reader,
            dna_file_path=Path(bpy.path.abspath(field(source, f"{component}_dna_file_path"))),
        )
        importer.rig_object = temporary
        importer.import_bones()
        rest = {bone.name: (bone.matrix_local.copy(), bone.parent.name if bone.parent else None) for bone in data.bones}
        replacement = rig.data.copy()
        temporary.data = replacement
        _activate(temporary)
        bpy.ops.object.mode_set(mode="EDIT")
        # Extra head roots were also authored in Maya axes. Artist-created bones
        # keep their existing axes and have identity animation conversion.
        from ..constants import EXTRA_BONES

        extras = {name for name, _ in EXTRA_BONES} if component == "head" else set()
        for name, (matrix, parent) in rest.items():
            bone = replacement.edit_bones.get(name) or replacement.edit_bones.new(name)
            bone.use_connect = False
            bone.length = bone.length or 1 / SCALE_FACTOR
            bone.matrix = matrix
            if parent:
                bone.parent = replacement.edit_bones.get(parent)
        for name in extras - rest.keys():
            bone = replacement.edit_bones.get(name)
            if bone:
                bone.use_connect = False
                bone.matrix = bone.matrix @ BASIS.inverted()
        bpy.ops.object.mode_set(mode="OBJECT")
        succeeded = True
        return replacement
    except Exception:
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        raise
    finally:
        bpy.data.objects.remove(temporary, do_unlink=True)
        bpy.data.armatures.remove(data)
        if replacement and replacement.users == 0 and not succeeded:
            bpy.data.armatures.remove(replacement)


def _bone_state(rig: Any) -> dict:
    return {
        bone.name: {
            "mode": bone.rotation_mode,
            "location": tuple(bone.location),
            "scale": tuple(bone.scale),
            "rotation_euler": tuple(bone.rotation_euler),
            "rotation_quaternion": tuple(bone.rotation_quaternion),
            "rotation_axis_angle": tuple(bone.rotation_axis_angle),
            "basis": bone.matrix_basis.copy(),
        }
        for bone in rig.pose.bones
    }


def _sample_basis(state: dict, curves: dict, prefix: str, frame: float) -> Matrix:
    def values(prop: str) -> list[float]:
        return [
            curves[(prefix + "." + prop, i)].evaluate(frame) if (prefix + "." + prop, i) in curves else value
            for i, value in enumerate(state[prop])
        ]

    mode = state["mode"]
    if mode == "QUATERNION":
        rotation = Quaternion(values("rotation_quaternion"))
        rotation.normalize()
    elif mode == "AXIS_ANGLE":
        angle, *axis = values("rotation_axis_angle")
        rotation = Quaternion(axis, angle)
    else:
        rotation = Euler(values("rotation_euler"), mode).to_quaternion()
    return Matrix.LocRotScale(values("location"), rotation, values("scale"))


def _write_curve(bag: Any, path: str, index: int, frames: list, values: list) -> None:
    curve = bag.fcurves.new(data_path=path, index=index)
    curve.keyframe_points.add(len(frames))
    curve.keyframe_points.foreach_set("co", [item for pair in zip(frames, values, strict=True) for item in pair])
    for point in curve.keyframe_points:
        point.interpolation = "LINEAR"
    curve.update()


def _permute_curves(bag: Any, prefix: str, mode: str) -> bool:
    """The usual 90-degree basis exchange can preserve authored curves exactly."""
    if mode != "QUATERNION":
        return False
    mapping = {
        "location": ((0, 1), (2, 1), (1, -1)),
        "scale": ((0, 1), (2, 1), (1, 1)),
        "rotation_quaternion": ((0, 1), (1, 1), (3, 1), (2, -1)),
    }
    curves = [(curve, prop) for curve in bag.fcurves for prop in mapping if curve.data_path == prefix + "." + prop]
    if any(curve.modifiers or curve.sampled_points for curve, _ in curves):
        return False
    # Temporarily move indices out of the way to avoid duplicate channel keys.
    originals = [(curve, prop, curve.array_index) for curve, prop in curves]
    for curve, _, index in originals:
        curve.array_index = index + 10
    for curve, prop, index in originals:
        new_index, sign = mapping[prop][index]
        curve.array_index = new_index
        for point in curve.keyframe_points:
            point.co.y *= sign
            point.handle_left.y *= sign
            point.handle_right.y *= sign
        curve.update()
    return True


def convert_action(action: Any, handle: int, rig: Any, state: dict, changes: dict, modes: dict) -> Any:
    """Copy one assigned slot; leave original Actions and other slots untouched."""
    copied = action.copy()
    copied.name = action.name + ".Migrated"
    copied.use_fake_user = True
    frames = sample_frames(action, handle, rig)
    try:
        bags = channel_containers(copied, handle)
        for bag in bags:
            curves = {(curve.data_path, curve.array_index): curve for curve in bag.fcurves}
            for name, change in changes.items():
                prefix = rig.pose.bones[name].path_from_id()
                paths = {
                    prefix + "." + prop
                    for prop in ("location", "scale", "rotation_euler", "rotation_quaternion", "rotation_axis_angle")
                }
                animated = [curve for curve in bag.fcurves if curve.data_path in paths]
                if not animated:
                    continue
                mode = modes[name]
                if (
                    mode == state[name]["mode"]
                    and _matrix_error(change, BASIS) < 1e-5
                    and _permute_curves(bag, prefix, mode)
                ):
                    continue
                inverse = change.inverted()
                samples = [change @ _sample_basis(state[name], curves, prefix, frame) @ inverse for frame in frames]
                for curve in animated:
                    bag.fcurves.remove(curve)
                locations, rotations, scales = [], [], []
                previous = None
                for matrix in samples:
                    location, quaternion, scale = matrix.decompose()
                    if mode == "QUATERNION":
                        if previous and quaternion.dot(previous) < 0:
                            quaternion.negate()
                        rotation = quaternion
                    else:
                        rotation = quaternion.to_euler(mode, previous) if previous else quaternion.to_euler(mode)
                    previous = rotation.copy()
                    locations.append(location)
                    rotations.append(rotation)
                    scales.append(scale)
                rotation_path = "rotation_quaternion" if mode == "QUATERNION" else "rotation_euler"
                for prop, samples in (("location", locations), (rotation_path, rotations), ("scale", scales)):
                    for index in range(len(samples[0])):
                        _write_curve(bag, prefix + "." + prop, index, frames, [sample[index] for sample in samples])
        copied["dna_migration_source"] = action.name
        return copied
    except Exception:
        bpy.data.actions.remove(copied)
        raise


def _matrix_error(left: Matrix, right: Matrix) -> float:
    return max(abs(left[row][column] - right[row][column]) for row in range(4) for column in range(4))


@dataclass
class RigUpgrade:
    rig: Any
    old_data: Any
    new_data: Any
    state: dict
    changes: dict
    modes: dict
    actions: dict
    bindings: list
    version: Any
    attachments: list
    applied: bool = False

    def apply(self, migrate_actions: bool) -> None:
        from ..dna_io.coordinates import COORDINATE_VERSION

        self.applied = True
        self.rig.data = self.new_data
        self.rig["dna_coordinate_version"] = COORDINATE_VERSION
        self.restore_pose()
        for assignment, action, handle, _muted in self.bindings:
            if not action.library:
                action.use_fake_user = True
            if migrate_actions:
                set_action(assignment, self.actions[(action, handle)], handle)
            elif isinstance(assignment, bpy.types.NlaStrip):
                assignment.mute = True
            else:
                set_action(assignment, None)
        for owner, attribute, inverse, change in self.attachments:
            setattr(owner, attribute, change @ inverse)

    def restore_pose(self) -> None:
        """Keep unkeyed input channels when initialization refreshes editor lists."""
        for name, change in self.changes.items():
            bone = self.rig.pose.bones[name]
            bone.rotation_mode = self.modes[name]
            bone.matrix_basis = change @ self.state[name]["basis"] @ change.inverted()
            if self.modes[name] == self.state[name]["mode"] == "QUATERNION" and _matrix_error(change, BASIS) < 1e-5:
                # Sparse quaternion curves depend on the unkeyed components too.
                # Matrix decomposition may choose the opposite quaternion sign.
                w, x, y, z = self.state[name]["rotation_quaternion"]
                bone.rotation_quaternion = (w, x, -z, y)
                x, y, z = self.state[name]["location"]
                bone.location = (x, -z, y)
                x, y, z = self.state[name]["scale"]
                bone.scale = (x, z, y)

    def rollback(self) -> None:
        if not self.applied:
            return
        self.rig.data = self.old_data
        if self.version is None:
            self.rig.pop("dna_coordinate_version", None)
        else:
            self.rig["dna_coordinate_version"] = self.version
        for name, state in self.state.items():
            bone = self.rig.pose.bones[name]
            bone.rotation_mode = state["mode"]
            for attribute in ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"):
                setattr(bone, attribute, state[attribute])
        for assignment, action, handle, muted in self.bindings:
            set_action(assignment, action, handle)
            if muted is not None:
                assignment.mute = muted
        for owner, attribute, inverse, _ in self.attachments:
            setattr(owner, attribute, inverse)

    def close(self, success: bool) -> None:
        unused = self.old_data if success else self.new_data
        if unused.users == 0:
            name = unused.name
            bpy.data.armatures.remove(unused)
            if success:
                self.new_data.name = name
        if not success:
            for action in self.actions.values():
                bpy.data.actions.remove(action)


def prepare(source: Any, migrate_actions: bool) -> list[RigUpgrade]:
    if not needs_coordinates(source):
        return []
    plans = []
    try:
        drivers = set()
        readers = {
            component: _reader(source, component)
            for component in ("head", "body")
            if field(source, component + "_rig") and field(source, component + "_dna_file_path")
        }
        # Head and body share neck/head quaternion input bones.
        for reader in readers.values():
            drivers.update(reader.getRawControlName(i).split(".")[0] for i in range(reader.getRawControlCount()))
        for component, rig in legacy_rigs(source):
            replacement = _stage_armature(source, component, rig, readers[component])
            state = _bone_state(rig)
            changes = {
                bone.name: replacement.bones[bone.name].matrix_local.inverted() @ bone.matrix_local
                for bone in rig.data.bones
            }
            modes = {
                name: "QUATERNION"
                if name in drivers or (component == "head" and not name.startswith("FACIAL_"))
                else "XYZ"
                for name in changes
            }
            plan = RigUpgrade(
                rig, rig.data, replacement, state, changes, modes, {}, [], rig.get("dna_coordinate_version"), []
            )
            plans.append(plan)
            for item in assignments(rig):
                plan.bindings.append((item, item.action, slot_handle(item), getattr(item, "mute", None)))
            if migrate_actions:
                for action, handle in assigned_actions(rig):
                    plan.actions[(action, handle)] = convert_action(action, handle, rig, state, changes, modes)
            for owner in bpy.context.scene.objects:
                if owner.parent == rig and owner.parent_type == "BONE" and owner.parent_bone in changes:
                    bone = rig.data.bones[owner.parent_bone]
                    new_bone = replacement.bones[owner.parent_bone]
                    if bone.use_relative_parent:
                        parent_change = Matrix.Identity(4)
                    else:
                        parent_change = (
                            Matrix.Translation((0, -new_bone.length, 0))
                            @ changes[bone.name]
                            @ Matrix.Translation((0, bone.length, 0))
                        )
                    plan.attachments.append(
                        (owner, "matrix_parent_inverse", owner.matrix_parent_inverse.copy(), parent_change)
                    )
                constraints = list(owner.constraints)
                if owner.pose:
                    constraints.extend(c for bone in owner.pose.bones for c in bone.constraints)
                for constraint in constraints:
                    if constraint.type == "CHILD_OF" and constraint.target == rig and constraint.subtarget in changes:
                        plan.attachments.append(
                            (
                                constraint,
                                "inverse_matrix",
                                constraint.inverse_matrix.copy(),
                                changes[constraint.subtarget],
                            )
                        )
        return plans
    except Exception:
        for plan in plans:
            plan.close(False)
        raise


# Optional control-rig rebuild


def _integration(context: Any, source: Any) -> tuple:
    if not hasattr(context.scene, "character_assembly"):
        raise ValueError("Enable Character Assembly for automatic control-rig migration")
    package = next((name for name in sys.modules if name.split(".")[-1] == "character_control_rig"), None)
    props = getattr(context.scene, "character_control_rig", None)
    if not package or props is None:
        raise ValueError("Enable Character Control Rig for automatic control-rig migration")
    utilities = importlib.import_module(package + ".utilities")
    name = field(source, "name") or field(source, "instance_name")
    proxy = utilities.get_rig_instance_proxy(props, name)
    if proxy is None:
        raise ValueError(f"{name}: no saved control-rig builder settings")
    builder = importlib.import_module(package + ".rig_builder").get_builder(proxy.framework)
    mappings = importlib.import_module(package + ".rig_builder.mappings")
    constants = importlib.import_module(package + ".constants")
    return builder, proxy, mappings, constants


def control_rebuild_available(context: Any, source: Any) -> bool:
    try:
        _integration(context, source)
    except (ImportError, ValueError, AttributeError, KeyError):
        return False
    return True


def _restore_pose(rig: Any, state: dict) -> None:
    for name, values in state.items():
        bone = rig.pose.bones.get(name)
        if bone:
            bone.rotation_mode = values["mode"]
            for attribute in ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"):
                setattr(bone, attribute, values[attribute])


def _control_motion_error(
    expected: Matrix, actual: Matrix, bone: Any, world_scale: Matrix
) -> tuple[float, float, float]:
    """Measure endpoint displacement in metres, rotation in radians, and scale.

    Skinning matrices act on bind-space points. Measuring the bone's head and
    tail avoids mistaking a rotation about a distant joint for a translation.
    ``world_scale`` includes the armature transform and scene unit scale.
    """
    position = max(
        (world_scale @ (actual @ point - expected @ point)).length for point in (bone.head_local, bone.tail_local)
    )
    rotation = expected.to_quaternion().rotation_difference(actual.to_quaternion())
    # atan2 preserves small angles that float32 acos(w) rounds to zero.
    angle = 2 * math.atan2(math.sqrt(rotation.x**2 + rotation.y**2 + rotation.z**2), abs(rotation.w))
    scale = max(abs(a - b) for a, b in zip(actual.to_scale(), expected.to_scale(), strict=True))
    return position, angle, scale


def _evaluate_action(rig: Any, action: Any, handle: int, callback: Any, frames: list | None = None) -> list:
    """Sample a single assigned Action independently of its NLA time mapping."""
    animation = rig.animation_data_create()
    previous, previous_handle = animation.action, slot_handle(animation)
    use_nla = animation.use_nla
    influence, blend = animation.action_influence, animation.action_blend_type
    state, object_basis = _bone_state(rig), rig.matrix_basis.copy()
    try:
        animation.use_nla = False
        animation.action_influence = 1
        animation.action_blend_type = "REPLACE"
        set_action(animation, action, handle)
        result = []
        for frame in frames if frames is not None else sample_frames(action, handle, rig):
            bpy.context.scene.frame_set(math.floor(frame), subframe=frame - math.floor(frame))
            bpy.context.view_layer.update()
            result.append(callback(frame))
        return result
    finally:
        set_action(animation, previous, previous_handle)
        animation.use_nla = use_nla
        animation.action_influence, animation.action_blend_type = influence, blend
        _restore_pose(rig, state)
        rig.matrix_basis = object_basis


def _rebake(action: Any, handle: int, owner: Any) -> Any:
    """Retain switches/custom channels and bake transform curves at scene frames."""
    result = action.copy()
    result.name = action.name + ".Migrated Controls"
    result.use_fake_user = True
    result["dna_migration_source"] = action.name
    frames = sample_frames(action, handle, owner)
    for bag in channel_containers(result, handle):
        for curve in list(bag.fcurves):
            if not curve.data_path.endswith((".location", ".rotation_euler", ".rotation_quaternion", ".scale")):
                continue
            values = [curve.evaluate(frame) for frame in frames]
            path, index = curve.data_path, curve.array_index
            bag.fcurves.remove(curve)
            _write_curve(bag, path, index, frames, values)
    return result


def _copy_nla(source: Any, destination: Any, copies: dict, migrate: bool) -> None:
    old = source.animation_data
    if not old:
        return
    new = destination.animation_data_create()
    for name in ("action_blend_type", "action_extrapolation", "action_influence", "use_nla"):
        setattr(new, name, getattr(old, name))
    if old.action and migrate:
        set_action(new, copies[(old.action, slot_handle(old))], slot_handle(old))
    for track in old.nla_tracks:
        target = new.nla_tracks.new()
        target.name, target.mute, target.is_solo = track.name, track.mute or not migrate, track.is_solo
        for strip in track.strips:
            # Complex strips are rejected before any mutation in prepare().
            key = (strip.action, slot_handle(strip))
            copied = target.strips.new(strip.name, int(strip.frame_start), copies[key] if migrate else strip.action)
            set_action(copied, copies[key] if migrate else strip.action, key[1])
            for name in (
                "action_frame_start",
                "action_frame_end",
                "scale",
                "repeat",
                "frame_start",
                "frame_end",
                "blend_type",
                "extrapolation",
                "influence",
                "blend_in",
                "blend_out",
                "use_auto_blend",
                "use_reverse",
                "use_sync_length",
                "mute",
            ):
                setattr(copied, name, getattr(strip, name))
        target.lock = track.lock


@dataclass
class ControlUpgrade:
    source: Any
    old: Any
    body: Any
    builder: Any
    proxy: Any
    mappings: Any
    constants: Any
    migrate: bool
    namespace: str
    name: str
    state: dict
    actions: dict = data_field(default_factory=dict)
    samples: dict = data_field(default_factory=dict)
    constraints: list = data_field(default_factory=list)
    new: Any = None
    metarig: Any = None
    applied: bool = False
    staged_objects: set = data_field(default_factory=set)

    @classmethod
    def prepare(cls, context: Any, source: Any, migrate: bool) -> ControlUpgrade:
        builder, proxy, mappings, constants = _integration(context, source)
        builder.validate_runtime(context)
        old, body = field(source, "control_rig"), field(source, "body_rig")
        if old.animation_data and old.animation_data.use_tweak_mode:
            raise ValueError(f"{old.name}: exit NLA Tweak Mode before migrating")
        if old.library or not old.is_editable:
            raise ValueError(f"{old.name}: migrate the control rig in its source .blend")
        if old.animation_data:
            for track in old.animation_data.nla_tracks:
                for strip in track.strips:
                    if (
                        strip.type != "CLIP"
                        or strip.use_animated_influence
                        or strip.use_animated_time
                        or strip.modifiers
                    ):
                        raise ValueError(
                            f"{old.name}: bake animated/meta NLA strips to Actions before rebuilding controls"
                        )
        plan = cls(
            {
                name: field(source, name)
                for name in ("name", "instance_name", "head_rig", "body_rig", "control_rig", "head_mesh", "body_mesh")
            },
            old,
            body,
            builder,
            proxy,
            mappings,
            constants,
            migrate,
            "DNA_upgrade_" + uuid.uuid4().hex[:8],
            field(source, "name") or field(source, "instance_name"),
            _bone_state(old),
        )
        deform = mappings.load_source_to_deform_mappings(builder.resource_id, constants.DEFAULT_CHARACTER)
        names = [pair["source"] for pair in deform if pair["source"] in body.pose.bones]
        rests = {name: body.data.bones[name].matrix_local.inverted() for name in names}

        def snapshot(frame: float) -> tuple:
            evaluated = body.evaluated_get(context.evaluated_depsgraph_get())
            return frame, {name: evaluated.pose.bones[name].matrix @ rests[name] for name in names}

        try:
            if migrate:
                for action, handle in assigned_actions(old):
                    plan.samples[(action, handle)] = _evaluate_action(old, action, handle, snapshot)
                    plan.actions[(action, handle)] = _rebake(action, handle, old)
            return plan
        except Exception:
            plan.close(False)
            raise

    def stage(self, body_data: Any) -> None:  # noqa: PLR0912
        before = {obj.as_pointer() for obj in bpy.data.objects}
        source_rig = self.body.copy()
        source_rig.data = body_data
        source_rig.animation_data_clear()
        bpy.context.scene.collection.objects.link(source_rig)
        for constraint in list(source_rig.constraints):
            source_rig.constraints.remove(constraint)
        for bone in source_rig.pose.bones:
            for constraint in list(bone.constraints):
                bone.constraints.remove(constraint)
            bone.matrix_basis = Matrix.Identity(4)
        # Builders enable follow controls on the selected character. Migration
        # must preserve those authored values, including on another character.
        follow = [
            (bone, bone.location.copy())
            for obj in bpy.context.scene.objects
            if obj.pose
            for name in ("CTRL_eyesAimFollowHead", "CTRL_faceGUIfollowHead")
            if (bone := obj.pose.bones.get(name))
        ]
        # Snap from rest positions, with posed meshes disabled for mesh-based snapping.
        rigs = [rig for component in ("head", "body") if (rig := field(self.source, component + "_rig"))]
        positions = [(rig.data, rig.data.pose_position) for rig in rigs]
        for data, _ in positions:
            data.pose_position = "REST"
        try:
            mappings = self.builder.load_snap_data(self.constants.DEFAULT_CHARACTER)
            kwargs = {}
            if self.proxy.snap_bones_to_mesh:
                kwargs = {
                    "body_mesh": field(self.source, "body_mesh"),
                    "head_mesh": field(self.source, "head_mesh"),
                    "bone_to_vertex_mappings": self.mappings.load_bone_to_vertex_mappings(
                        self.builder.resource_id, self.constants.DEFAULT_CHARACTER
                    ),
                }
            bpy.context.view_layer.update()
            self.metarig = self.builder.create_metarig(self.namespace, source_rig, mappings, **kwargs)
            self.new = self.builder.generate_control_rig(self.namespace, self.metarig)
            self.metarig = None  # Builders consume their metarig on success.
            if self.new is None:
                raise RuntimeError(f"{self.name}: control-rig generation failed")
            self.new.parent = self.old.parent
            self.new.parent_type = self.old.parent_type
            self.new.parent_bone = self.old.parent_bone
            self.new.matrix_parent_inverse = self.old.matrix_parent_inverse.copy()
            self.new.matrix_world = self.old.matrix_world.copy()
            for name, value in self.old.items():
                if name != "rig_id":
                    self.new[name] = value
            _restore_pose(self.new, self.state)
            for bone in self.old.pose.bones:
                target = self.new.pose.bones.get(bone.name)
                if target:
                    for name, value in bone.items():
                        target[name] = value
            # A custom metarig cannot be reconstructed from standard DNA positions.
            # Reject before replacement instead of silently losing authored motion.
            if self.migrate:
                for action, handle in assigned_actions(self.old):
                    for bag in channel_containers(action, handle):
                        for curve in bag.fcurves:
                            try:
                                self.new.path_resolve(curve.data_path)
                            except (ValueError, KeyError) as error:
                                raise ValueError(f"{action.name}: rebuilt controls lack {curve.data_path}") from error
            _copy_nla(self.old, self.new, self.actions, self.migrate)
        finally:
            bpy.data.objects.remove(source_rig, do_unlink=True)
            for bone, location in follow:
                bone.location = location
            self.staged_objects = {obj.as_pointer() for obj in bpy.data.objects} - before
            for data, position in positions:
                data.pose_position = position

    def apply(self) -> None:
        self.applied = True
        prefix = self.constants.CONSTRAINT_PREFIX
        self.constraints = [
            (bone, constraint, constraint.mute)
            for bone in self.body.pose.bones
            for constraint in bone.constraints
            if constraint.name.startswith(prefix)
        ]
        for _, constraint, _ in self.constraints:
            constraint.mute = True
        mappings = self.mappings.load_source_to_deform_mappings(
            self.builder.resource_id, self.constants.DEFAULT_CHARACTER
        )
        self.builder.constrain_to_source(self.body, self.new, mappings, self.namespace)

    def verify(self) -> None:
        for key, expected in self.samples.items():
            action, handle = self.actions[key], key[1]
            names = expected[0][1]
            rests = {name: self.body.data.bones[name].matrix_local.inverted() for name in names}

            def snapshot(frame: float, names: Any = names, rests: Any = rests) -> tuple:
                evaluated = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get())
                world_scale = evaluated.matrix_world.to_3x3() * bpy.context.scene.unit_settings.scale_length
                return frame, {name: evaluated.pose.bones[name].matrix @ rests[name] for name in names}, world_scale

            actual = _evaluate_action(self.new, action, handle, snapshot, [frame for frame, _ in expected])
            worst = (0.0, 0.0, "", (0.0, 0.0, 0.0))
            for (frame, values, world_scale), (_, old_values) in zip(actual, expected, strict=True):
                for name in names:
                    errors = _control_motion_error(
                        old_values[name], values[name], self.body.data.bones[name], world_scale
                    )
                    score = max(
                        error / limit if math.isfinite(error) else math.inf
                        for error, limit in zip(errors, _CONTROL_MOTION_LIMITS, strict=True)
                    )
                    if score > worst[0]:
                        worst = (score, frame, name, errors)
            score, frame, name, (position, angle, scale) = worst
            if score > 1:
                raise ValueError(
                    f"{key[0].name}: rebuilt control rig changes {name} at frame {frame:g} "
                    f"({position * 1000:.3f} mm, {math.degrees(angle):.3f} degrees, {scale * 100:.3f}% scale). "
                    "Limits: 0.1 mm, 0.1 degrees, 0.01% scale. Original rig and actions retained."
                )

    def rollback(self) -> None:
        if not self.applied:
            return
        old_constraints = {constraint.as_pointer() for _, constraint, _ in self.constraints}
        for bone in self.body.pose.bones:
            for constraint in list(bone.constraints):
                if (
                    constraint.name.startswith(self.constants.CONSTRAINT_PREFIX)
                    and constraint.as_pointer() not in old_constraints
                ):
                    bone.constraints.remove(constraint)
        for _, constraint, mute in self.constraints:
            constraint.mute = mute

    def close(self, success: bool) -> None:  # noqa: PLR0912
        if success:
            for item in assignments(self.old):
                if not item.action.library:
                    item.action.use_fake_user = True
            old_name = self.old.name
            collections = list(self.old.users_collection)
            for collection in list(self.new.users_collection):
                collection.objects.unlink(self.new)
            for collection in collections:
                collection.objects.link(self.new)
            # Update assembly attachments, driver targets and the instance pointer
            # before releasing the superseded control object.
            self.old.user_remap(self.new)
            for bone, constraint, _ in self.constraints:
                bone.constraints.remove(constraint)
            collection_name = self.name + "_" + self.constants.CONSTRAINTS_COLLECTION_SUFFIX
            old_collection = bpy.data.collections.get(collection_name)
            if old_collection:
                for obj in list(old_collection.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(old_collection)
            self.builder.remove(None, self.old, None, self.name)
            self.new.name = old_name
            self.proxy.metarig = None
            for collection in bpy.data.collections:
                if collection.name.startswith(self.namespace + "_"):
                    collection.name = self.name + collection.name[len(self.namespace) :]
        else:
            self.builder.remove(None, self.new, self.metarig, self.namespace)
            # Builders can fail after producing a partial rig. All of these IDs
            # were created in this staging call; the user's old rig is untouched.
            for obj in list(bpy.data.objects):
                if obj.as_pointer() in self.staged_objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            collection = bpy.data.collections.get(self.namespace + "_" + self.constants.CONSTRAINTS_COLLECTION_SUFFIX)
            if collection:
                for obj in list(collection.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(collection)
            for action in self.actions.values():
                bpy.data.actions.remove(action)
