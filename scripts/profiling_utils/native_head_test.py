"""Validate graph-driven native head evaluation against the Python reference."""

from __future__ import annotations

import argparse
import json
import math
import sys

from pathlib import Path
from typing import Any

import bpy

from mathutils import Quaternion


def bind_output_copies(instance: Any, candidate: bpy.types.Object, binding: dict) -> tuple[list, bpy.types.Material]:
    """Bind independent output copies so native and reference never share writers."""
    targets = []
    key_targets = []
    for key_blocks, positions, channels, _blocks, _buffer in instance.head_shape_key_apply_plan:
        source = next(
            obj for obj in bpy.data.objects if obj.type == "MESH" and obj.data.shape_keys == key_blocks.id_data
        )
        duplicate = source.copy()
        duplicate.data = source.data.copy()
        duplicate.name = f"Native_{source.name}"
        bpy.context.scene.collection.objects.link(duplicate)
        for modifier in duplicate.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == instance.head_rig:
                modifier.object = candidate
        key_targets.append(
            {
                "key": duplicate.data.shape_keys,
                "positions": [int(value) for value in positions],
                "channels": [int(value) for value in channels],
            }
        )
        targets.append((source, duplicate))
    material = instance.head_material.copy()
    binding["keys"] = key_targets
    binding["material"] = material
    binding["texture_node"] = instance.head_texture_masks_node.name
    node = material.node_tree.nodes[binding["texture_node"]]
    sockets = {socket.name: index for index, socket in enumerate(node.inputs)}
    binding["maps"] = [
        {"channel": channel, "socket": sockets[name]}
        for channel, name in instance.head_animated_map_plan
        if name in sockets
    ]
    for _source, duplicate in targets:
        for slot in duplicate.material_slots:
            if slot.material == instance.head_material:
                slot.material = material
    return targets, material


def head_binding(instance: Any) -> dict:
    """Serialize the existing head mapping once, outside native evaluation."""
    reader = instance.head_dna_reader
    variable = {
        int(attribute) // 9
        for group in range(reader.getJointGroupCount())
        for attribute in reader.getJointGroupOutputIndices(group)
    }
    joints = []
    for index, bone, location, rotation, scale, inverse, has_children in instance.head_bone_transform_plan:
        joints.append(
            {
                "index": index,
                "bone": bone,
                "location": list(location),
                "rotation": list(rotation),
                "scale": list(scale),
                "rest_inverse": [value for row in inverse for value in row],
                "constant": int(index not in variable),
                "has_children": int(has_children),
            }
        )
    return {
        "path": bpy.path.abspath(instance.head_dna_file_path),
        "lod": 0,
        "joints": joints,
        "drivers": [{"bone": name} for name in sorted(instance.head_driver_bone_names)],
        "face_board": instance.face_board,
        "gui": [
            {"bone": name, "channel": index, "axis": "xyz".index(axis)}
            for index, name, axis in instance.head_gui_control_plan
        ],
    }


def run(fixture: Path, output: Path, eye_aim: bool = False) -> None:  # noqa: PLR0912, PLR0915
    """Compare local and posed matrices with no Python writer on the native rig."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from profiling_utils.ci_benchmark import setup_environment
    from profiling_utils.native_body_test import _assert_finite_matrix

    bpy.ops.wm.open_mainfile(filepath=str(fixture))
    if not setup_environment():
        raise RuntimeError("Reference add-on unavailable")
    from character_dna import rig_instance
    from character_dna.utilities import get_active_rig_instance, get_addon_window_manager_properties

    instance = get_active_rig_instance()
    instance.initialize()
    get_addon_window_manager_properties().evaluate_dependency_graph = False
    for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.frame_change_post):
        for handler in list(handlers):
            if getattr(handler, "__module__", "") == rig_instance.__name__:
                handlers.remove(handler)
    if bpy.app.timers.is_registered(rig_instance.run_main_thread_evaluations):
        bpy.app.timers.unregister(rig_instance.run_main_thread_evaluations)

    reference = instance.head_rig
    candidate = reference.copy()
    candidate.data = reference.data.copy()
    candidate.name = "AdaNativeHead"
    bpy.context.scene.collection.objects.link(candidate)
    binding = head_binding(instance)
    binding["debug_raw"] = [0.0] * instance.head_dna_reader.getRawControlCount()
    binding["debug_gui"] = [0.0] * instance.head_dna_reader.getGUIControlCount()
    binding["debug_eye"] = [0.0, 0.0, 0.0]
    targets, material = bind_output_copies(instance, candidate, binding)
    candidate["_riglogic_head"] = binding
    names = [joint["bone"] for joint in binding["joints"]]
    original = {name: candidate.pose.bones[name].matrix_basis.copy() for name in names}
    results = []
    for case in range(10):
        instance.face_board.pose.bones["CTRL_lookAtSwitch"].location.y = float(eye_aim)
        if eye_aim:
            instance.face_board.pose.bones["CTRL_eyesAimFollowHead"].location.y = float(case % 2)
            instance.face_board.pose.bones["CTRL_faceGUIfollowHead"].location.y = float(case % 2)
            instance.face_board.pose.bones["CTRL_C_eyesAim"].location.x = math.sin(case) * 0.1
        for index, name, axis in instance.head_gui_control_plan:
            bone = instance.face_board.pose.bones.get(name)
            if bone:
                setattr(bone.location, axis, math.sin(index * 0.7 + case) * 0.3 if case else 0.0)
        center = instance.face_board.pose.bones.get("CTRL_C_eye")
        if center:
            center.location.x = 0.25 if case % 2 else 0.0
        for index, name in enumerate(sorted(instance.head_driver_bone_names)):
            bone = instance.body_rig.pose.bones[name]
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = Quaternion((0, 1, 0), math.sin(case + index) * 0.15 if case else 0.0)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        instance.apply_dependency_graph_update(graph)
        instance.update_head_switch_values()
        for _iteration in range(12 if eye_aim else 1):
            bpy.context.view_layer.update()
            graph = bpy.context.evaluated_depsgraph_get()
            instance.apply_dependency_graph_update(graph)
            instance.update_head_gui_control_values(dependency_graph=graph)
            instance.update_head_bone_transforms()
            instance.update_head_shape_keys()
            instance.update_head_texture_masks()
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = candidate.evaluated_get(graph)
        expected = reference.evaluated_get(graph)
        eye_stats = list(evaluated["_riglogic_head"]["debug_eye"])
        if eye_aim and (eye_stats[2] != 1.0 or eye_stats[0] > 13 or eye_stats[1] > 1e-6):
            raise AssertionError(f"Native eye aim failed to converge: {eye_stats}")
        native_raw = list(evaluated["_riglogic_head"]["debug_raw"])
        reference_raw = list(instance.head_instance.getRawControlValues())
        raw_error = max(
            abs(actual - expected_value) for actual, expected_value in zip(native_raw, reference_raw, strict=True)
        )
        worst_raw = max(range(len(native_raw)), key=lambda index: abs(native_raw[index] - reference_raw[index]))
        print(
            f"HEAD INPUT case={case} eye_aim={instance.head_use_eye_aim} raw_error={raw_error} "
            f"control={instance.head_dna_reader.getRawControlName(worst_raw)} "
            f"native={native_raw[worst_raw]} python={reference_raw[worst_raw]}",
            flush=True,
        )
        maximum_local = maximum_pose = original_error = 0.0
        worst_bone = ""
        for name in names:
            actual_local = evaluated.pose.bones[name].matrix_basis
            expected_local = reference.pose.bones[name].matrix_basis
            actual_pose = evaluated.pose.bones[name].matrix
            expected_pose = expected.pose.bones[name].matrix
            for matrix in (actual_local, expected_local, actual_pose, expected_pose):
                _assert_finite_matrix(matrix, name)
            local_error = max(
                abs(actual_local[row][column] - expected_local[row][column]) for row in range(4) for column in range(4)
            )
            if local_error > maximum_local:
                maximum_local, worst_bone = local_error, name
            maximum_pose = max(
                maximum_pose,
                *(
                    abs(actual_pose[row][column] - expected_pose[row][column])
                    for row in range(4)
                    for column in range(4)
                ),
            )
            original_error = max(
                original_error,
                *(
                    abs(candidate.pose.bones[name].matrix_basis[row][column] - original[name][row][column])
                    for row in range(4)
                    for column in range(4)
                ),
            )
        vertex_error = 0.0
        for source, duplicate in targets:
            expected_vertices = source.evaluated_get(graph).data.vertices
            actual_vertices = duplicate.evaluated_get(graph).data.vertices
            if len(expected_vertices) != len(actual_vertices):
                raise AssertionError("Evaluated vertex counts differ")
            for expected_vertex, actual_vertex in zip(expected_vertices, actual_vertices, strict=True):
                for axis in range(3):
                    if not math.isfinite(actual_vertex.co[axis]):
                        raise AssertionError("Non-finite native vertex")
                    vertex_error = max(vertex_error, abs(expected_vertex.co[axis] - actual_vertex.co[axis]))
        native_node = material.evaluated_get(graph).node_tree.nodes[binding["texture_node"]]
        mask_error = max(
            abs(
                native_node.inputs[item["socket"]].default_value
                - instance.head_texture_masks_node.inputs[item["socket"]].default_value
            )
            for item in binding["maps"]
        )
        passed = (
            maximum_local <= 1e-5
            and maximum_pose <= 1e-5
            and original_error == 0.0
            and vertex_error <= 1e-5
            and mask_error <= 1e-5
        )
        results.append(
            {
                "case": case,
                "max_local_error": maximum_local,
                "max_pose_error": maximum_pose,
                "original_error": original_error,
                "max_vertex_error": vertex_error,
                "max_mask_error": mask_error,
                "eye_stats": eye_stats,
                "worst_bone": worst_bone,
                "passed": passed,
            }
        )
        print(f"HEAD OUTPUT vertices={vertex_error:.9g} masks={mask_error:.9g}", flush=True)
        print(
            f"HEAD PARITY case={case} local={maximum_local:.9g} pose={maximum_pose:.9g} bone={worst_bone}", flush=True
        )
    report = {
        "scope": "native_head_output_parity",
        "eye_aim_fixed_point": eye_aim,
        "binary": bpy.app.binary_path,
        "joints": len(names),
        "constant_joints": sum(joint["constant"] for joint in binding["joints"]),
        "cases": results,
        "passed": all(case["passed"] for case in results),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if not report["passed"]:
        raise AssertionError(f"Native head parity failed: {output}")
    print(f"NATIVE HEAD PASS: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eye-aim", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.fixture, args.output, args.eye_aim)
