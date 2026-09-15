"""Explicit editor and bake sampling through the same native frame evaluator."""

from __future__ import annotations

from array import array
from typing import Any

import bpy

from mathutils import Euler, Vector

from . import engine, frame


def _record(instance: Any, component: str) -> dict[str, Any]:
    key = instance.cache_key(component, "runtime_plan")
    record = instance.data.get(key)
    if record is not None:
        return record
    available, reason = engine.capability()
    if not available:
        raise RuntimeError(reason)
    plans = getattr(instance, f"{component}_bone_transform_plan")
    raw = instance.head_raw_quat_plan if component == "head" else instance.body_raw_plan
    carrier = {
        "component": component,
        "rig": getattr(instance, f"{component}_rig"),
        "face": instance.face_board if component == "head" else None,
        "plan": engine.transform_plan(plans, component),
        "raw": [{"index": index, "name": name, "axis": axis} for index, name, axis in raw],
        "gui": [{"index": index, "name": name, "axis": axis} for index, name, axis in instance.head_gui_control_plan]
        if component == "head"
        else [],
        "joint_count": len(plans),
        "joints": [{"name": item[1]} for item in plans],
    }
    record = engine.make_record(carrier, instance)
    context = frame.buffers(record)
    context.update(
        session=engine.native_module().create_session(record["model"]),
        sdk=array("d", [0.0]) * sum(record["info"]["output_counts"]),
        local=array("d", [0.0]) * (record["joint_count"] * 9),
    )
    record["sample"] = context
    instance.data[key] = record
    return record


def sample(instance: Any, component: str, overrides: dict | None = None, graph: Any = None) -> Any:
    """Sample live or overridden GUI inputs without writing scene outputs."""
    record = _record(instance, component)
    context = record["sample"]
    graph = graph or bpy.context.evaluated_depsgraph_get()
    aim = frame.capture(record, context, graph)
    if overrides and record["face"]:
        for name, values in overrides.items():
            index = record["face"].pose.bones.find(name)
            if index < 0:
                continue
            for axis, value in values.items():
                if axis in ("x", "y", "z"):
                    context["face"][index * 3 + "xyz".index(axis)] = value
        switch = overrides.get("CTRL_lookAtSwitch", {}).get("y")
        if switch is not None:
            aim = switch >= 0.99
    native = engine.native_module()
    lod = min(int(instance.view_options.active_lod[-1]), record["info"]["lod_count"] - 1)
    native.evaluate_frame(
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
        aim,
    )
    controls = native.control_snapshot(context["session"])
    target = instance.data[instance.cache_key(component, "instance")]
    for index, value in enumerate(controls["gui"]):
        target.setGUIControl(index, value)
    for index, value in enumerate(controls["raw"]):
        target.setRawControl(index, value)
    target.setLOD(lod)
    if component == "body" and overrides:
        for index, name, axis in record["raw"]:
            value = overrides.get(name, {}).get(axis)
            if value is not None:
                target.setRawControl(index, value)
    getattr(instance, f"{component}_manager").calculate(target)
    engine.mark_authoring_current(instance, component)
    return target


def raw_inputs(instance: Any, component: str, overrides: dict | None = None) -> None:
    """Update joint-driving controls through the native input-capture plan."""
    if component == "body":
        sample(instance, component, overrides)
        return
    target = instance.head_instance
    reader = instance.head_dna_reader
    previous_raw = array("f", (target.getRawControl(index) for index in range(reader.getRawControlCount())))
    previous_gui = array("f", (target.getGUIControl(index) for index in range(reader.getGUIControlCount())))
    sample(instance, component)
    for index, name, axis in instance.head_raw_quat_plan:
        if overrides is None:
            previous_raw[index] = target.getRawControl(index)
        else:
            previous_raw[index] = overrides.get(name, {}).get(axis, previous_raw[index])
    for index, value in enumerate(previous_gui):
        target.setGUIControl(index, value)
    for index, value in enumerate(previous_raw):
        target.setRawControl(index, value)
    instance.head_manager.calculate(target)


def bone_transforms(instance: Any, component: str, collect: bool = False) -> list:
    """Convert SDK authoring outputs in C++, applying only outside native ownership."""
    if not getattr(instance, f"{component}_rig") or not getattr(instance, f"{component}_dna_reader"):
        return []
    record = _record(instance, component)
    state = getattr(instance, f"{component}_instance")
    values = record["sample"]["local"]
    engine.native_module().transform_into(
        record["plan"], array("d", state.getJointOutputs()), values, component == "body"
    )
    bound = engine.active(instance) and instance.auto_evaluate and getattr(instance, f"auto_evaluate_{component}")
    if bound:
        engine.preview_controls(instance, state, component)
    result = []
    rig = getattr(instance, f"{component}_rig")
    for name, slot in record["bone_slots"].items():
        offset = slot * 9
        location = Vector(values[offset : offset + 3])
        rotation = Euler(values[offset + 3 : offset + 6], "XYZ")
        scale = Vector(values[offset + 6 : offset + 9])
        if not bound:
            bone = rig.pose.bones[name]
            bone.location, bone.rotation_euler, bone.scale = location, rotation, scale
        if collect:
            result.append((name, location, rotation, scale))
    return result


def evaluate_once(instance: Any, component: str) -> None:
    """Apply one native sample while automatic output drivers are disabled."""
    if component == "head" and instance.face_board:
        targets, switches, visibility_start, _values = engine.switch_targets(instance)
        for index, (target, switch) in enumerate(zip(targets, switches, strict=True)):
            value = instance.face_board.pose.bones[switch].location.y
            path, _, property_name = target.path.rpartition(".")
            owner = target.owner.path_resolve(path)
            setattr(owner, property_name, value < 0.99 if index >= visibility_start else value)
        bpy.context.view_layer.update()
    sample(instance, component)
    if instance.evaluate_bones:
        bone_transforms(instance, component)
    if component == "head":
        if instance.evaluate_shape_keys:
            instance.update_head_shape_keys()
        if instance.evaluate_texture_masks:
            instance.update_head_texture_masks()
    getattr(instance, f"{component}_rig").update_tag()
    bpy.context.view_layer.update()
