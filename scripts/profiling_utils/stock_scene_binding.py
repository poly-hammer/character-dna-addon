"""Disposable real-scene carrier binding used to validate graph topology first."""

from __future__ import annotations

import time
import uuid

from array import array
from typing import Any

import bpy


def _variable(driver: Any, name: str, owner: Any, path: str) -> None:
    variable = driver.variables.new()
    variable.name = name
    variable.type = "SINGLE_PROP"
    variable.targets[0].id_type = owner.id_type
    variable.targets[0].id = owner
    variable.targets[0].data_path = path


def _identity(owner: Any, path: str, index: int | None = None) -> Any:
    curve = owner.driver_add(path) if index is None else owner.driver_add(path, index)
    curve.keyframe_points.clear()
    for modifier in tuple(curve.modifiers):
        curve.modifiers.remove(modifier)
    return curve.driver


def bind_scene(instance: Any, native: Any, array_trigger: bool = False) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
    """Install test-only drivers; do not save or use as production lifecycle code."""
    from character_dna import utilities
    from profiling_utils.native_frame_benchmark import _remove_python_evaluation

    if instance.head_use_eye_aim:
        raise ValueError("Topology probe currently requires eye aim OFF")
    instance.evaluate()
    _remove_python_evaluation()
    bpy.context.evaluated_depsgraph_get()
    records: dict[str, Any] = {}
    errors: list[str] = []
    installed: list[Any] = []
    for component in ("body", "head"):
        rig = getattr(instance, f"{component}_rig")
        model = native.load_model(
            bpy.path.abspath(getattr(instance, f"{component}_dna_file_path")), component == "body", True
        )
        session = native.create_session(model)
        info = native.describe(session)
        plan = getattr(instance, f"{component}_bone_transform_plan")
        active = set(getattr(instance, f"{component}_manager").getJointVariableAttributeIndices(0))
        dynamic_plan = (
            plan
            if component == "body"
            else [item for item in plan if any(item[0] * 9 + axis in active for axis in range(9))]
        )
        carrier = bpy.data.objects.new(f"RigLogicProbe_{component}", None)
        bpy.context.scene.collection.objects.link(carrier)
        token = uuid.uuid4().hex
        carrier["token"] = token
        carrier["epoch"] = 0.0
        size = len(dynamic_plan) * 9 + sum(info["output_counts"][1:])
        carrier["outputs"] = array("d", [-1000.0]) * (size + int(array_trigger))
        record = {
            "model": model,
            "info": info,
            "rig": rig,
            "carrier": carrier,
            "plan": dynamic_plan,
            "component": component,
            "contexts": {},
            "calls": 0,
            "stage_seconds": [0.0, 0.0, 0.0],
            "native_plan": array(
                "f",
                [
                    value
                    for item in dynamic_plan
                    for value in (
                        item[0],
                        int(item[6]) if component == "head" else 0,
                        *item[2],
                        *item[3],
                        *item[4],
                        *(value for row in item[5] for value in row),
                    )
                ],
            ),
        }
        records[token] = record
        for output_index, item in enumerate(dynamic_plan):
            bone = rig.pose.bones[item[1]]
            if bone.rotation_mode != "XYZ":
                raise ValueError(f"Non-XYZ output bone {component}:{bone.name}")
            for offset, prop in enumerate(("location", "rotation_euler", "scale")):
                for axis in range(3):
                    path = bone.path_from_id(prop)
                    if rig.animation_data and rig.animation_data.drivers.find(path, index=axis):
                        raise ValueError(f"Existing output driver: {path}[{axis}]")
                    driver = _identity(bone, prop, axis)
                    if not array_trigger:
                        _variable(driver, "epoch", carrier, '["epoch"]')
                    _variable(driver, "output", carrier, f'["outputs"][{output_index * 9 + offset * 3 + axis}]')
                    driver.expression = "output" if array_trigger else "output + 0 * epoch"
                    installed.append(driver)
        if component == "head":
            shapes_offset = len(dynamic_plan) * 9
            for blocks, positions, channels, _driven, _buffer in instance.head_shape_key_apply_plan:
                for position, channel in zip(positions, channels, strict=True):
                    block = blocks[int(position)]
                    driver = _identity(block, "value")
                    if not array_trigger:
                        _variable(driver, "epoch", carrier, '["epoch"]')
                    _variable(driver, "output", carrier, f'["outputs"][{shapes_offset + int(channel)}]')
                    driver.expression = "output" if array_trigger else "output + 0 * epoch"
                    installed.append(driver)
            maps_offset = shapes_offset + info["output_counts"][1]
            final_maps = {name: channel for channel, name in instance.head_animated_map_plan}
            for name, channel in final_maps.items():
                socket = instance.head_texture_masks_node.inputs.get(name)
                if socket is None:
                    continue
                driver = _identity(socket, "default_value")
                if not array_trigger:
                    _variable(driver, "epoch", carrier, '["epoch"]')
                _variable(driver, "output", carrier, f'["outputs"][{maps_offset + channel}]')
                driver.expression = "output" if array_trigger else "output + 0 * epoch"
                installed.append(driver)

    def solve(owner: Any, dependency_graph: Any) -> float:
        try:
            record = records[owner["token"]]
            if not owner.is_evaluated or owner == record["carrier"]:
                raise RuntimeError("Expected evaluated scratch carrier")
            record["calls"] += 1
            started = time.perf_counter()
            key = (dependency_graph.as_pointer(), owner.as_pointer(), dependency_graph.mode)
            context = record["contexts"].get(key)
            if context is None:
                context = {
                    "session": native.create_session(record["model"]),
                    "raw": array("f", [0.0])
                    * (
                        len(record["info"]["raw_names"])
                        + (len(record["info"]["gui_names"]) if record["component"] == "head" else 0)
                    ),
                    "native_outputs": array("d", [0.0]) * sum(record["info"]["output_counts"]),
                    "local_outputs": array("d", [0.0]) * (len(record["plan"]) * 9),
                }
                record["contexts"][key] = context
            component = record["component"]
            evaluated_rig = record["rig"].evaluated_get(dependency_graph)
            raw = context["raw"]
            raw_offset = len(record["info"]["gui_names"]) if component == "head" else 0
            if component == "head":
                face = instance.face_board.evaluated_get(dependency_graph)
                for index, name, axis in instance.head_gui_control_plan:
                    bone = face.pose.bones.get(name)
                    value = getattr(bone.location, axis) if bone else 0.0
                    if name in {"CTRL_L_eye", "CTRL_R_eye"}:
                        center = face.pose.bones.get("CTRL_C_eye")
                        if center and abs(getattr(center.location, axis)) > 0.0001:
                            value = getattr(center.location, axis)
                    raw[index] = value
            converted = {}
            raw_plan = instance.body_raw_plan if component == "body" else instance.head_raw_quat_plan
            for index, name, axis in raw_plan:
                if name not in converted:
                    converted[name] = utilities.get_pose_bone_local_quaternion(evaluated_rig.pose.bones[name])
                raw[raw_offset + index] = getattr(converted[name], axis)
            captured = time.perf_counter()
            native.evaluate_into(
                context["session"],
                raw,
                context["native_outputs"],
                int(instance.view_options.active_lod[-1]),
                False,
                component == "head",
            )
            solved = time.perf_counter()
            outputs = context["native_outputs"]
            transformed = context["local_outputs"]
            native.transform_into(record["native_plan"], outputs, transformed, component == "body")
            destination = memoryview(owner["outputs"])
            if array_trigger:
                destination = destination[:-1]
            destination[: len(transformed)] = memoryview(transformed)
            destination[len(transformed) :] = memoryview(outputs)[record["info"]["output_counts"][0] :]
            record["stage_seconds"][0] += captured - started
            record["stage_seconds"][1] += solved - captured
            record["stage_seconds"][2] += time.perf_counter() - solved
            return float(record["calls"])
        except Exception as error:
            errors.append(str(error))
            raise

    bpy.app.driver_namespace["rl_scene_probe"] = solve
    for record in records.values():
        driver = (
            _identity(record["carrier"], '["outputs"]', len(record["carrier"]["outputs"]) - 1)
            if array_trigger
            else _identity(record["carrier"], '["epoch"]')
        )
        driver.use_self = True
        names = sorted(getattr(instance, f"{record['component']}_driver_bone_names"))
        for index, name in enumerate(names):
            for suffix, bone_name in (
                ("bone", name),
                (
                    "parent",
                    record["rig"].pose.bones[name].parent.name if record["rig"].pose.bones[name].parent else name,
                ),
            ):
                variable = driver.variables.new()
                variable.name = f"input_{index}_{suffix}"
                variable.type = "TRANSFORMS"
                variable.targets[0].id = record["rig"]
                variable.targets[0].bone_target = bone_name
                variable.targets[0].transform_type = "LOC_X"
                variable.targets[0].transform_space = "WORLD_SPACE"
        if record["component"] == "head":
            for index, (_channel, name, axis) in enumerate(instance.head_gui_control_plan):
                bone = instance.face_board.pose.bones.get(name)
                if bone:
                    _variable(
                        driver,
                        f"gui_{index}",
                        instance.face_board,
                        bone.path_from_id("location") + f"[{'xyz'.index(axis)}]",
                    )
            for axis in range(2):
                _variable(
                    driver,
                    f"center_{axis}",
                    instance.face_board,
                    instance.face_board.pose.bones["CTRL_C_eye"].path_from_id("location") + f"[{axis}]",
                )
        driver.expression = "rl_scene_probe(self, depsgraph)"
        installed.append(driver)
    return {"records": records, "errors": errors, "drivers": installed}
