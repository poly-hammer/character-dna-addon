"""Native model ownership, input capture and evaluated carrier publication."""

from __future__ import annotations

import hashlib
import logging
import sys
import uuid

from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import bpy

from ..constants import ToolInfo
from .bindings import (
    CARRIER_MARKER,
    Target,
    add_variable,
    install_targets,
    owned_curve,
    remove_targets,
    validate_targets,
)


logger = logging.getLogger(__name__)
NAMESPACE = "character_dna_native_solve_v1"
SCHEMA_VERSION = 2
SETTINGS = (
    "auto_evaluate",
    "auto_evaluate_head",
    "auto_evaluate_body",
    "evaluate_bones",
    "evaluate_rbfs",
    "evaluate_shape_keys",
    "evaluate_texture_masks",
)
_records: dict[int, dict[str, Any]] = {}
_models: dict[tuple[str, bool], Any] = {}
_module: Any = None
_generation = 0
_errors: dict[str, str] = {}
_warnings: dict[int, str] = {}
MIGRATION_WARNING = "Legacy data detected. Run Migrate Legacy Data, then save the file."


def capability() -> tuple[bool, str]:
    """Check availability without importing private Blender APIs or development paths."""
    global _module
    if bpy.app.version < (4, 5, 0) or sys.version_info[:2] not in {(3, 11), (3, 13)}:
        return False, "RigLogic requires Blender 4.5 or newer with supported Python bindings"
    if not bpy.context.preferences.filepaths.use_scripts_auto_execute and bpy.app.autoexec_fail:
        return False, "This file has Python driver execution disabled"
    try:
        if _module is None:
            from ..bindings import load_native_runtime

            _module = load_native_runtime()
        info = _module.capabilities()
        if info["api_version"] != 1 or info["private_blender_api"]:
            return False, "Incompatible native runtime module"
        for name in ("evaluate_frame", "create_frame_plan", "load_model", "create_session", "control_snapshot"):
            if not callable(getattr(_module, name, None)):
                return False, f"Native runtime is missing {name}"
    except (ImportError, OSError, RuntimeError) as error:
        return False, f"Native runtime unavailable: {error}"
    return True, "Native runtime available"


def _model(path: str, body: bool) -> Any:
    resolved = Path(bpy.path.abspath(path)).resolve(strict=True)
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    key = (digest, body)
    if key not in _models:
        _models[key] = _module.load_model(str(resolved), body, True)
    return _models[key]


def instance_id(instance: Any) -> str:
    """Return the persisted runtime identity without using mutable display names."""
    return instance.get("native_runtime_id", "")


def carriers(instance: Any = None) -> list[Any]:
    """Resolve original carrier IDs afresh after undo, remapping or loading."""
    return [
        obj
        for obj in bpy.data.objects
        if obj.get(CARRIER_MARKER) == 1 and (instance is None or _belongs_to(obj, instance))
    ]


def _belongs_to(carrier: Any, instance: Any) -> bool:
    component = carrier.get("component")
    if component not in {"head", "body", "switches"}:
        return False
    field = "face_board" if component == "switches" else f"{component}_rig"
    owner = getattr(instance, field, None) if hasattr(instance, "bl_rna") else instance.get(field)
    if isinstance(owner, str):
        owner = bpy.data.objects.get(owner)
    return owner is not None and carrier.get("face" if component == "switches" else "rig") == owner


def active(instance: Any) -> bool:
    """Whether native drivers own any output channels for the instance."""
    return any(_belongs_to(record["carrier"], instance) for record in _records.values())


def bound_carriers(instance: Any) -> list[Any]:
    """Read active carrier ownership without traversing scene objects from the UI."""
    return [record["carrier"] for record in _records.values() if _belongs_to(record["carrier"], instance)]


def invalidate() -> None:
    """Release borrowed wrappers and sessions before graph/ID storage can disappear."""
    from .ui_refresh import invalidate_migration

    invalidate_migration()
    global _generation
    _generation += 1
    _records.clear()
    _models.clear()
    _warnings.clear()


def clear_contexts() -> None:
    """Retire completed graph sessions without accessing Blender IDs."""
    global _generation
    _generation += 1
    for record in _records.values():
        record["contexts"].clear()
        record.pop("last_context", None)


def discard(instance: Any = None) -> None:
    """Remove only owned drivers and carriers in a write-safe operator/lifecycle context."""
    from .ui_refresh import invalidate_migration

    invalidate_migration()
    for carrier in carriers(instance):
        if carrier.library or carrier.override_library:
            continue
        remove_targets(carrier)
        _errors.pop(carrier.get("instance_id", ""), None)
        _records.pop(carrier.as_pointer(), None)
        _warnings.pop(carrier.as_pointer(), None)
        bpy.data.objects.remove(carrier, do_unlink=True)
    if not _records:
        _models.clear()
        _errors.clear()


def release_records(instance: Any = None) -> None:
    """Release sessions without removing or modifying persisted driver bindings."""
    if instance is None:
        invalidate()
        return
    for carrier in carriers(instance):
        _records.pop(carrier.as_pointer(), None)
        _warnings.pop(carrier.as_pointer(), None)
    _warnings.pop(instance.as_pointer(), None)
    _errors.pop(instance_id(instance), None)


def _resolve_instance(carrier: Any, scene: Any = None) -> Any:
    scenes = [scene] if scene is not None else bpy.data.scenes
    if not carrier.library:
        for candidate in scenes:
            if candidate.library:
                continue
            properties = getattr(candidate, ToolInfo.NAME, None)
            for instance in getattr(properties, "rig_instance_list", []):
                if _belongs_to(carrier, instance):
                    return instance
    settings = dict(carrier.get("settings", {}))
    values: dict[str, Any] = {flag: bool(settings.get(flag, True)) for flag in SETTINGS}
    values["view_options"] = SimpleNamespace(active_lod=settings.get("active_lod", "lod0"))
    for component in ("head", "body"):
        path = carrier.get(f"{component}_dna_file_path", "")
        values[f"{component}_dna_file_path"] = bpy.path.abspath(path, library=carrier.library) if path else ""
    values["get"] = lambda key, default=None: carrier.get("instance_id" if key == "native_runtime_id" else key, default)
    values["name"] = carrier.get("character_name", "")
    return SimpleNamespace(**values)


def sync_settings(instance: Any, _context: Any = None) -> None:
    """Publish local evaluation settings without introducing a Scene dependency."""
    for carrier in carriers(instance):
        if carrier.library or carrier.override_library:
            continue
        settings = {flag: bool(getattr(instance, flag)) for flag in SETTINGS}
        settings["active_lod"] = instance.view_options.active_lod
        if dict(carrier.get("settings", {})) != settings:
            carrier["settings"] = settings
            carrier.update_tag()


def _describe_carrier(carrier: Any, instance: Any) -> None:
    carrier["schema_version"] = SCHEMA_VERSION
    carrier["character_name"] = instance.name
    for component in ("head", "body"):
        path = getattr(instance, f"{component}_dna_file_path")
        carrier[f"{component}_dna_file_path"] = bpy.path.abspath(path) if path else ""
    carrier["settings"] = {flag: bool(getattr(instance, flag)) for flag in SETTINGS}
    carrier["settings"]["active_lod"] = instance.view_options.active_lod


def carrier_issues(carrier: Any) -> list[str]:
    """Validate a saved graph without installing drivers or requiring live sessions."""
    issues = []
    if carrier.get("schema_version") != SCHEMA_VERSION:
        return [f"{carrier.name}: incompatible native binding schema; run Migrate Legacy Data"]
    if carrier.get("scene") is not None:
        issues.append(f"{carrier.name}: obsolete source scene dependency")
    if not carrier.is_property_overridable_library('["targets"]'):
        issues.append(f"{carrier.name}: native targets require migration for editable linking")
    animation = carrier.animation_data
    epoch = animation.drivers.find('["epoch"]') if animation else None
    if epoch is None or NAMESPACE not in epoch.driver.expression:
        issues.append(f"{carrier.name}: missing native solve driver")
    for target in carrier.get("targets", []):
        owner = target.get("owner")
        if owner and target.get("embedded"):
            owner = owner.node_tree
        curve = (
            owner.animation_data.drivers.find(target["path"], index=max(0, target["index"]))
            if (owner and owner.animation_data)
            else None
        )
        if curve is None or not owned_curve(curve, carrier):
            issues.append(f"{carrier.name}: missing or conflicting output driver {target['path']}")
    return issues


def requires_migration(carrier: Any) -> bool:
    """Recognize older persisted layouts, not failed sessions or corrupt current bindings."""
    version = carrier.get("schema_version", 1)
    return isinstance(version, int) and (
        version < SCHEMA_VERSION
        or (
            version == SCHEMA_VERSION
            and (carrier.get("scene") is not None or not carrier.is_property_overridable_library('["targets"]'))
        )
    )


def binding_issues(instance: Any) -> list[str]:
    """Report persisted runtime compatibility, independent of current evaluation state."""
    found = carriers(instance)
    issues = [issue for carrier in found for issue in carrier_issues(carrier)]
    issues.extend(
        f"Missing persisted {component} native runtime"
        for component in ("head", "body")
        if getattr(instance, f"{component}_rig")
        and instance.auto_evaluate
        and getattr(instance, f"auto_evaluate_{component}")
        and not any(carrier.get("component") == component for carrier in found)
    )
    return issues


def adopt(instance: Any = None) -> int:
    """Hydrate compatible saved bindings without writing their original datablocks."""
    from .ui_refresh import invalidate_migration

    invalidate_migration()
    available, reason = capability()
    if not available:
        raise RuntimeError(reason)
    found = carriers(instance)
    issues = (
        binding_issues(instance)
        if instance is not None
        else [issue for carrier in found for issue in carrier_issues(carrier)]
    )
    if issues:
        raise ValueError("; ".join(issues))
    bpy.app.driver_namespace[NAMESPACE] = solve
    count = 0
    for carrier in found:
        key = carrier.as_pointer()
        if key not in _records:
            _records[key] = make_record(carrier, instance)
        _warnings.pop(key, None)
        count += 1
    if instance is not None:
        _warnings.pop(instance.as_pointer(), None)
    return count


def native_module() -> Any:
    """Return the required, already validated native module."""
    return _module


def transform_plan(plans: list, component: str) -> array:
    """Pack the common C++ transform plan for live evaluation and authoring."""
    return array(
        "f",
        [
            value
            for item in plans
            for value in (
                item[0],
                int(item[6]) if component == "head" else 0,
                *item[2],
                *item[3],
                *item[4],
                *(value for row in item[5] for value in row),
            )
        ],
    )


def make_record(carrier: Any, instance: Any = None) -> dict[str, Any]:
    """Build a native frame plan from persisted bindings or an authoring descriptor."""
    instance = instance if instance is not None else _resolve_instance(carrier)
    component = carrier["component"]
    if component == "switches":
        return {
            "identity": instance_id(instance),
            "instance": instance,
            "carrier": carrier,
            "component": component,
            "face": carrier["face"],
            "contexts": {},
            "calls": 0,
            "switches": list(carrier["switches"]),
            "visibility_start": carrier.get("visibility_start", len(carrier["switches"])),
        }
    model = _model(getattr(instance, f"{component}_dna_file_path"), component == "body")
    info = _module.describe(_module.create_session(model))
    record = {
        "identity": instance_id(instance),
        "instance": instance,
        "carrier": carrier,
        "rig": carrier["rig"],
        "face": carrier.get("face"),
        "model": model,
        "info": info,
        "component": component,
        "contexts": {},
        "plan": array("f", carrier["plan"]),
        "gui": [(item["index"], item["name"], item["axis"]) for item in carrier.get("gui", [])],
        "raw": [(item["index"], item["name"], item["axis"]) for item in carrier["raw"]],
        "joint_count": carrier["joint_count"],
        "calls": 0,
        "bone_slots": {item["name"]: index for index, item in enumerate(carrier["joints"])},
        "input_bones": {item["name"] for item in carrier["raw"]},
    }
    from .frame import prepare

    record["frame_plan"] = prepare(record, _module)
    return record


def restore() -> None:
    """Reconstruct native sessions from saved carrier metadata without changing IDs."""
    invalidate()
    hydrate()


def hydrate() -> None:
    """Adopt newly loaded compatible carriers without retiring existing sessions."""
    if not capability()[0]:
        return
    bpy.app.driver_namespace[NAMESPACE] = solve
    for carrier in carriers():
        try:
            if carrier.as_pointer() in _records:
                continue
            if requires_migration(carrier):
                warn_unavailable(carrier, MIGRATION_WARNING)
                continue
            issues = carrier_issues(carrier)
            if issues:
                raise ValueError("; ".join(issues))
            record = make_record(carrier)
            _records[carrier.as_pointer()] = record
            _warnings.pop(carrier.as_pointer(), None)
            _errors.pop(carrier.get("instance_id", ""), None)
        except FileNotFoundError:
            warn_unavailable(carrier, "DNA file not found. Update the DNA file path, then rebuild evaluation.")
        except Exception as error:
            _errors[carrier.get("instance_id", "")] = str(error)
            logger.exception("Unable to restore native rig runtime")


def _link_carrier(instance: Any, carrier: bpy.types.Object) -> None:
    scene = instance.id_data
    parent = instance.get("reference_root") or bpy.data.collections.get((instance.name, None))
    if parent is None:
        parent = bpy.data.collections.new(instance.name)
    if parent not in scene.collection.children_recursive:
        scene.collection.children.link(parent)
    name = f"{instance.name}_drivers"
    collection = bpy.data.collections.get((name, None))
    if collection is None:
        collection = bpy.data.collections.new(name)
    if collection not in parent.children.values():
        parent.children.link(collection)
    if collection in scene.collection.children.values():
        scene.collection.children.unlink(collection)
    collection.objects.link(carrier)


def install(instance: Any) -> None:  # noqa: PLR0912, PLR0915
    """Bind initialized rig data; roll back the whole instance on failure."""
    from .ui_refresh import invalidate_migration

    invalidate_migration()
    available, reason = capability()
    if not available:
        raise RuntimeError(reason)
    if active(instance) or carriers(instance):
        raise RuntimeError("Rig already has native bindings; remove them before rebuilding")
    if not instance.auto_evaluate:
        return
    identity = instance_id(instance) or uuid.uuid4().hex
    instance["native_runtime_id"] = identity
    prepared = []
    all_targets = []
    for component in ("body", "head"):
        rig = getattr(instance, f"{component}_rig")
        if not rig or not getattr(instance, f"auto_evaluate_{component}"):
            continue
        if rig.mode == "EDIT" or rig.library or not rig.is_editable:
            raise ValueError(f"Rig is not writable in object or pose mode: {rig.name}")
        model = _model(getattr(instance, f"{component}_dna_file_path"), component == "body")
        info = _module.describe(_module.create_session(model))
        plans = getattr(instance, f"{component}_bone_transform_plan")
        if component == "head":
            manager = instance.head_manager
            indices = set().union(
                *(set(manager.getJointVariableAttributeIndices(lod)) for lod in range(info["lod_count"]))
            )
            plans = [item for item in plans if any(item[0] * 9 + axis in indices for axis in range(9))]
        targets = []
        initial = array("d")
        for output_index, item in enumerate(plans):
            bone = rig.pose.bones[item[1]]
            if bone.rotation_mode != "XYZ":
                raise ValueError(f"Native output bone must use XYZ rotation: {bone.name}")
            for offset, property_name in enumerate(("location", "rotation_euler", "scale")):
                for axis in range(3):
                    targets.append(
                        Target(rig, bone.path_from_id(property_name), axis, output_index * 9 + offset * 3 + axis)
                    )
                    initial.append(getattr(bone, property_name)[axis])
        shapes_offset = len(initial)
        initial.extend([0.0] * sum(info["output_counts"][1:]))
        if component == "head":
            for blocks, positions, channels, _driven, _buffer in instance.head_shape_key_apply_plan:
                for position, channel in zip(positions, channels, strict=True):
                    block = blocks[int(position)]
                    targets.append(Target(block.id_data, block.path_from_id("value"), -1, shapes_offset + int(channel)))
                    initial[shapes_offset + int(channel)] = block.value
            maps_offset = shapes_offset + info["output_counts"][1]
            node = instance.head_texture_masks_node
            for name, channel in {name: channel for channel, name in instance.head_animated_map_plan}.items():
                socket = node.inputs.get(name) if node else None
                if socket:
                    targets.append(
                        Target(socket.id_data, socket.path_from_id("default_value"), -1, maps_offset + channel)
                    )
                    initial[maps_offset + channel] = socket.default_value
        all_targets.extend(targets)
        prepared.append((component, rig, info, plans, targets, initial))
    validate_targets(all_targets)
    try:
        for component, rig, _info, plans, targets, initial in prepared:
            carrier = bpy.data.objects.new(f"{instance.name}_{component}_native", None)
            carrier[CARRIER_MARKER] = 1
            carrier["token"] = uuid.uuid4().hex
            carrier["instance_id"] = identity
            _describe_carrier(carrier, instance)
            carrier["rig"] = rig
            carrier["component"] = component
            carrier["joint_count"] = len(plans)
            carrier["joints"] = [{"name": item[1]} for item in plans]
            carrier["outputs"] = initial
            carrier["epoch"] = 0.0
            carrier["preview_revision"] = 0
            carrier["plan"] = transform_plan(plans, component)
            raw_plan = instance.body_raw_plan if component == "body" else instance.head_raw_quat_plan
            carrier["raw"] = [{"index": index, "name": name, "axis": axis} for index, name, axis in raw_plan]
            if component == "head" and instance.face_board:
                carrier["face"] = instance.face_board
                carrier["gui"] = [
                    {"index": index, "name": name, "axis": axis} for index, name, axis in instance.head_gui_control_plan
                ]
            _link_carrier(instance, carrier)
            carrier.hide_select = True
            carrier.empty_display_size = 0.001
            install_targets(carrier, targets)
            carrier.property_overridable_library_set('["targets"]', True)
            curve = carrier.driver_add('["epoch"]')
            assert isinstance(curve, bpy.types.FCurve)
            curve.keyframe_points.clear()
            for modifier in tuple(curve.modifiers):
                curve.modifiers.remove(modifier)
            driver = curve.driver
            assert driver is not None
            driver.use_self = True
            add_variable(driver, "preview_revision", carrier, '["preview_revision"]')
            for index, name in enumerate(sorted({name for _index, name, _axis in raw_plan})):
                bone = rig.pose.bones.get(name)
                if not bone:
                    raise ValueError(f"Missing quaternion input bone: {name}")
                for suffix, target in (("bone", bone), ("parent", bone.parent or bone)):
                    variable = driver.variables.new()
                    variable.name = f"input_{index}_{suffix}"
                    variable.type = "TRANSFORMS"
                    variable.targets[0].id = rig
                    variable.targets[0].bone_target = target.name
                    variable.targets[0].transform_type = "LOC_X"
                    variable.targets[0].transform_space = "WORLD_SPACE"
            for index, item in enumerate(carrier.get("gui", [])):
                face = instance.face_board
                bone = face.pose.bones.get(item["name"])
                if bone:
                    add_variable(
                        driver, f"gui_{index}", face, bone.path_from_id("location") + f"[{'xyz'.index(item['axis'])}]"
                    )
            if component == "head" and instance.face_board:
                center = instance.face_board.pose.bones.get("CTRL_C_eye")
                if center:
                    for axis in range(2):
                        add_variable(
                            driver, f"center_{axis}", instance.face_board, center.path_from_id("location") + f"[{axis}]"
                        )
                switch = instance.face_board.pose.bones.get("CTRL_lookAtSwitch")
                if switch:
                    add_variable(driver, "eye_mode", instance.face_board, switch.path_from_id("location") + "[1]")
                for index, name in enumerate(("CTRL_L_eyeAim", "CTRL_R_eyeAim")):
                    if instance.face_board.pose.bones.get(name):
                        variable = driver.variables.new()
                        variable.name = f"eye_target_{index}"
                        variable.type = "TRANSFORMS"
                        variable.targets[0].id = instance.face_board
                        variable.targets[0].bone_target = name
                        variable.targets[0].transform_type = "LOC_X"
                        variable.targets[0].transform_space = "WORLD_SPACE"
            for flag in SETTINGS:
                add_variable(driver, flag, carrier, f'["settings"]["{flag}"]')
            driver.expression = f"{NAMESPACE}(self, depsgraph)"
            _records[carrier.as_pointer()] = make_record(carrier, instance)
            _warnings.pop(carrier.as_pointer(), None)
        if instance.head_rig and instance.face_board and instance.auto_evaluate_head:
            _install_switches(instance, identity)
        bpy.app.driver_namespace[NAMESPACE] = solve
        _errors.pop(identity, None)
        _warnings.pop(instance.as_pointer(), None)
    except Exception:
        discard(instance)
        raise


def solve(owner: Any, graph: Any) -> float:
    """Publish only the current evaluated carrier buffer, never original scene IDs."""
    from .frame import buffers, capture

    original = owner.original
    record = _records.get(original.as_pointer())
    if record is None:
        warn_unavailable(original)
        return float(owner.get("epoch", 0.0))
    try:
        if not owner.is_evaluated or owner == record["carrier"]:
            raise RuntimeError("Native output publication requires evaluated carrier storage")
        if graph.view_layer.objects.get(original.name) != original:
            return float(owner.get("epoch", 0.0))
        instance = _resolve_instance(record["carrier"], graph.scene)
        component = record["component"]
        if component == "switches":
            face = record["face"].evaluated_get(graph)
            values = array(
                "d",
                [
                    float(face.pose.bones[name].location.y < 0.99)
                    if index >= record["visibility_start"]
                    else face.pose.bones[name].location.y
                    for index, name in enumerate(record["switches"])
                ],
            )
            memoryview(owner["outputs"])[:] = memoryview(values)
            record["calls"] += 1
            return float(record["calls"])
        rig = record["rig"].evaluated_get(graph)
        key = (_generation, graph.as_pointer(), owner.as_pointer(), graph.mode)
        context = record["contexts"].get(key)
        info = record["info"]
        if context is None:
            context = {
                "session": _module.create_session(record["model"]),
                "sdk": array("d", [0.0]) * sum(info["output_counts"]),
                "local": array("d", [0.0]) * (record["joint_count"] * 9),
                "published": array("d", record["carrier"]["outputs"]),
            }
            context.update(buffers(record))
            record["contexts"][key] = context
        if (
            not instance.auto_evaluate
            or not getattr(instance, f"auto_evaluate_{component}")
            or rig.data.pose_position == "REST"
        ):
            memoryview(owner["outputs"])[:] = memoryview(context["published"])
            return 0.0
        eye_aim = capture(record, context, graph)
        lod = min(int(instance.view_options.active_lod[-1]), info["lod_count"] - 1)
        _module.evaluate_frame(
            context["session"],
            record["frame_plan"],
            context["poses"],
            context["bases"],
            context["face"],
            context["spatial"],
            context["sdk"],
            context["local"],
            lod,
            instance.evaluate_rbfs,
            eye_aim,
        )
        published = memoryview(context["published"])
        count = record["joint_count"] * 9
        if instance.evaluate_bones:
            published[:count] = memoryview(context["local"])
        shapes = info["output_counts"][1]
        offset = info["output_counts"][0]
        if component == "head" and instance.evaluate_shape_keys:
            published[count : count + shapes] = memoryview(context["sdk"])[offset : offset + shapes]
        if component == "head" and instance.evaluate_texture_masks:
            published[count + shapes :] = memoryview(context["sdk"])[offset + shapes :]
        _apply_preview(record, context, owner, graph, instance)
        memoryview(owner["outputs"])[:] = published
        record["calls"] += 1
        context["revision"] = record["calls"]
        context["lod"] = lod
        record["last_context"] = context
        return float(record["calls"])
    except Exception as error:
        _errors[record["identity"]] = str(error)
        raise


def status(instance: Any) -> str:
    """Expose binding failures instead of silently reporting stale output as success."""
    warning = next(
        (_warnings[owner.as_pointer()] for owner in [instance, *carriers(instance)] if owner.as_pointer() in _warnings),
        None,
    )
    return _errors.get(instance_id(instance), warning or ("Native" if active(instance) else "Inactive"))


def warn_unavailable(carrier: Any, message: str | None = None) -> None:
    """Warn once per carrier and reason while expected runtime setup is unavailable."""
    key = carrier.as_pointer()
    if message is None:
        if key in _warnings:
            return
        message = (
            MIGRATION_WARNING
            if requires_migration(carrier)
            else "Native evaluation is not initialized. Rebuild Native Evaluation."
        )
    if _warnings.get(key) != message:
        _warnings[key] = message
        logger.warning("%s: %s", carrier.name, message)


def errors() -> list[str]:
    """Return current evaluation failures for the preferences status display."""
    return list(_errors.values())


def _head_record(instance: Any) -> dict[str, Any] | None:
    return next(
        (
            record
            for record in _records.values()
            if _belongs_to(record["carrier"], instance) and record["component"] == "head"
        ),
        None,
    )


def _graph_context(record: dict[str, Any], graph: Any) -> dict[str, Any]:
    carrier = record["carrier"].evaluated_get(graph)
    key = (_generation, graph.as_pointer(), carrier.as_pointer(), graph.mode)
    return record["contexts"].get(key, {})


def _store_preview(record: dict[str, Any], outputs: array, controls: dict[str, Any], shape_channel: int = -1) -> None:
    graph = bpy.context.evaluated_depsgraph_get()
    context = _graph_context(record, graph)
    source = _module.control_snapshot(context["session"])
    carrier = record["carrier"]
    carrier["preview"] = {
        "outputs": outputs,
        "raw": array("f", controls["raw"]),
        "gui": array("f", controls["gui"]),
        "source_raw": array("f", source["raw"]),
        "source_gui": array("f", source["gui"]),
        "frame": graph.scene_eval.frame_current_final,
        "lod": context["lod"],
        "shape_channel": shape_channel,
    }
    carrier["preview_revision"] = carrier.get("preview_revision", 0) + 1
    curve = carrier.animation_data.drivers.find('["epoch"]')
    expression = f"{NAMESPACE}(self, depsgraph) + 0 * frame"
    if curve.driver.expression != expression:
        curve.driver.expression = expression
    carrier.update_tag()


def preview_controls(instance: Any, state: Any, component: str = "head") -> bool:
    """Publish a calculated manual head pose through the existing native drivers."""
    record = next(
        (
            item
            for item in _records.values()
            if _belongs_to(item["carrier"], instance) and item["component"] == component
        ),
        None,
    )
    if record is None:
        return False
    reader = getattr(instance, f"{component}_dna_reader")
    controls = {
        "raw": array("f", (state.getRawControl(index) for index in range(reader.getRawControlCount()))),
        "gui": array("f", (state.getGUIControl(index) for index in range(reader.getGUIControlCount()))),
    }
    outputs = array("d", [0.0]) * (record["joint_count"] * 9)
    _module.transform_into(record["plan"], array("d", state.getJointOutputs()), outputs, component == "body")
    outputs.extend(state.getBlendShapeOutputs())
    outputs.extend(state.getAnimatedMapOutputs())
    _store_preview(record, outputs, controls)
    return True


def preview_shape(instance: Any, channel: int, value: float) -> bool:
    """Override one shape channel without changing the currently previewed raw pose."""
    record = _head_record(instance)
    if record is None:
        return False
    graph = bpy.context.evaluated_depsgraph_get()
    context = _graph_context(record, graph)
    outputs = array("d", context["published"])
    outputs[record["joint_count"] * 9 + channel] = value
    controls = context.get("preview_controls") or _module.control_snapshot(context["session"])
    _store_preview(record, outputs, controls, shape_channel=channel)
    return True


def _apply_preview(record: dict[str, Any], context: dict[str, Any], owner: Any, graph: Any, instance: Any) -> None:
    context.pop("preview_controls", None)
    revision = owner.get("preview_revision", 0)
    if revision != record.get("preview_revision", 0):
        record["preview_revision"] = revision
        data = owner.get("preview")
        record["preview"] = (
            {
                **{name: array("f", data[name]) for name in ("raw", "gui", "source_raw", "source_gui")},
                "outputs": array("d", data["outputs"]),
                "frame": data["frame"],
                "lod": data["lod"],
                "shape_channel": data.get("shape_channel", -1),
            }
            if data
            else None
        )
    preview = record.get("preview")
    if preview is None:
        return
    matches = (
        graph.scene_eval.frame_current_final == preview["frame"]
        and int(instance.view_options.active_lod[-1]) == preview["lod"]
    )
    if matches:
        source = _module.control_snapshot(context["session"])
        matches = source["raw"] == preview["source_raw"] and source["gui"] == preview["source_gui"]
    if not matches:
        if graph.mode == "VIEWPORT":
            record["preview"] = None
        return
    count = record["joint_count"] * 9
    shapes = record["info"]["output_counts"][1]
    published = memoryview(context["published"])
    values = memoryview(preview["outputs"])
    for enabled, start, end in (
        (instance.evaluate_bones, 0, count),
        (instance.evaluate_shape_keys, count, count + shapes),
        (instance.evaluate_texture_masks, count + shapes, len(published)),
    ):
        if enabled:
            published[start:end] = values[start:end]
    if preview["shape_channel"] >= 0:
        index = count + preview["shape_channel"]
        published[index] = values[index]
    context["preview_controls"] = preview


def ui_raw_control_value(instance: Any, index: int, graph: Any) -> float | None:
    """Read the visible graph's native controls without solving or syncing the legacy SDK."""
    identity = instance_id(instance)
    if not identity:
        return None
    for record in _records.values():
        if not _belongs_to(record["carrier"], instance) or record["component"] != "head":
            continue
        carrier = record["carrier"].evaluated_get(graph)
        key = (_generation, graph.as_pointer(), carrier.as_pointer(), graph.mode)
        context = record["contexts"].get(key)
        if context is None or "revision" not in context:
            return 0.0
        revision = context["revision"]
        if context.get("ui_raw_revision") != revision:
            controls = context.get("preview_controls") or _module.control_snapshot(context["session"])
            context["ui_raw_values"] = controls["raw"]
            context["ui_raw_revision"] = revision
        values = context["ui_raw_values"]
        return float(values[index]) if 0 <= index < len(values) else 0.0
    return None


def mark_authoring_current(instance: Any, component: str) -> None:
    """Keep an explicit tool calculation from being overwritten by a repeated getter."""
    for record in _records.values():
        if _belongs_to(record["carrier"], instance) and record["component"] == component:
            record["authoring_revision"] = record["calls"]


def synchronize_authoring(instance: Any, component: str, state: Any) -> Any:
    """Populate SDK editing handles on demand, never as a live evaluation backend."""
    identity = instance_id(instance)
    if state is None or not identity:
        return state
    for record in _records.values():
        if not _belongs_to(record["carrier"], instance) or record["component"] != component:
            continue
        context = record.get("last_context")
        if context is None or record.get("authoring_revision") == record["calls"]:
            return state
        controls = context.get("preview_controls") or _module.control_snapshot(context["session"])
        for index, value in enumerate(controls["gui"]):
            state.setGUIControl(index, value)
        for index, value in enumerate(controls["raw"]):
            state.setRawControl(index, value)
        state.setLOD(context["lod"])
        manager = instance.data.get(instance.cache_key(component, "manager"))
        manager.calculate(state)
        record["authoring_revision"] = record["calls"]
        return state
    return state


def _install_switches(instance: Any, identity: str) -> None:
    face = instance.face_board
    targets = []
    switches = []
    values = array("d")
    for bone_name, switch_name in (
        ("CTRL_faceGUI", "CTRL_faceGUIfollowHead"),
        ("CTRL_C_eyesAim", "CTRL_eyesAimFollowHead"),
    ):
        bone = face.pose.bones.get(bone_name)
        switch = face.pose.bones.get(switch_name)
        constraint = next((item for item in bone.constraints if item.type == "CHILD_OF"), None) if bone else None
        if constraint and switch:
            targets.append(Target(face, constraint.path_from_id("influence"), -1, len(switches)))
            switches.append(switch_name)
            values.append(constraint.influence)
    visibility_start = len(switches)
    aim = face.pose.bones.get("CTRL_C_eyesAim")
    if aim and face.pose.bones.get("CTRL_lookAtSwitch"):
        for bone in [aim, *aim.children_recursive]:
            if bone != aim and bone.name.startswith(("GRP_", "LOC_")):
                continue
            visibility = bone if hasattr(bone, "hide") else bone.bone
            targets.append(Target(visibility.id_data, visibility.path_from_id("hide"), -1, len(switches)))
            switches.append("CTRL_lookAtSwitch")
            values.append(float(visibility.hide))
    if not targets:
        return
    validate_targets(targets)
    carrier = bpy.data.objects.new(f"{instance.name}_switches_native", None)
    carrier[CARRIER_MARKER] = 1
    carrier["token"] = uuid.uuid4().hex
    carrier["instance_id"] = identity
    _describe_carrier(carrier, instance)
    carrier["component"] = "switches"
    carrier["face"] = face
    carrier["switches"] = switches
    carrier["visibility_start"] = visibility_start
    carrier["epoch"] = 0.0
    carrier["outputs"] = values
    _link_carrier(instance, carrier)
    carrier.hide_select = True
    carrier.empty_display_size = 0.001
    install_targets(carrier, targets)
    carrier.property_overridable_library_set('["targets"]', True)
    curve = carrier.driver_add('["epoch"]')
    assert isinstance(curve, bpy.types.FCurve)
    curve.keyframe_points.clear()
    for modifier in tuple(curve.modifiers):
        curve.modifiers.remove(modifier)
    assert curve.driver is not None
    curve.driver.use_self = True
    for index, name in enumerate(switches):
        add_variable(curve.driver, f"switch_{index}", face, face.pose.bones[name].path_from_id("location") + "[1]")
    curve.driver.expression = f"{NAMESPACE}(self, depsgraph)"
    _records[carrier.as_pointer()] = make_record(carrier, instance)
