"""Compare native graph-driven Ada body outputs to the existing Python mapper."""

from __future__ import annotations

import argparse
import json
import math
import sys

from pathlib import Path

import bpy

from mathutils import Matrix, Quaternion


def _assert_finite_matrix(matrix: Matrix, label: str) -> None:
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise AssertionError(f"Non-finite body matrix: {label}")


def run(fixture: Path, output: Path) -> None:
    """Run numerical parity without a Python callback on the native body rig."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from profiling_utils.ci_benchmark import setup_environment

    bpy.ops.wm.open_mainfile(filepath=str(fixture))
    if not setup_environment():
        raise RuntimeError("Reference add-on is unavailable")
    from character_dna import rig_instance
    from character_dna.utilities import get_active_rig_instance, get_addon_window_manager_properties

    instance = get_active_rig_instance()
    if instance is None:
        raise RuntimeError("No reference rig instance in fixture")
    instance.initialize()
    get_addon_window_manager_properties().evaluate_dependency_graph = False
    for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.frame_change_post):
        for handler in list(handlers):
            if getattr(handler, "__module__", "") == rig_instance.__name__:
                handlers.remove(handler)
    if bpy.app.timers.is_registered(rig_instance.run_main_thread_evaluations):
        bpy.app.timers.unregister(rig_instance.run_main_thread_evaluations)

    reference = instance.body_rig
    candidate = reference.copy()
    candidate.data = reference.data.copy()
    candidate.name = "AdaNativeBody"
    bpy.context.scene.collection.objects.link(candidate)
    joints = []
    for index, bone, location, rotation, scale, inverse in instance.body_bone_transform_plan:
        joints.append(
            {
                "index": index,
                "bone": bone,
                "location": list(location),
                "rotation": list(rotation),
                "scale": list(scale),
                "rest_inverse": [value for row in inverse for value in row],
            }
        )
    candidate["_riglogic_body"] = {
        "path": bpy.path.abspath(instance.body_dna_file_path),
        "lod": 0,
        "joints": joints,
        "drivers": [{"bone": name} for name in sorted(instance.body_driver_bone_names)],
    }
    original_basis = {
        joint["bone"]: [value for row in candidate.pose.bones[joint["bone"]].matrix_basis for value in row]
        for joint in joints
    }
    results = []
    for case in range(8):
        for index, name in enumerate(sorted(instance.body_driver_bone_names)):
            axis = ((index % 3) == 0, (index % 3) == 1, (index % 3) == 2)
            angle = math.sin(index * 0.3 + case) * 0.15 if case else 0.0
            rotation = Quaternion(axis, angle)
            for rig in (reference, candidate):
                rig.pose.bones[name].rotation_mode = "QUATERNION"
                rig.pose.bones[name].rotation_quaternion = rotation
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        instance.apply_dependency_graph_update(graph)
        instance.update_body_raw_control_values()
        instance.update_body_bone_transforms()
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = candidate.evaluated_get(graph)
        expected = reference.evaluated_get(graph)
        maximum_local = 0.0
        maximum_pose = 0.0
        original_error = 0.0
        for joint in joints:
            name = joint["bone"]
            for matrix in (
                evaluated.pose.bones[name].matrix_basis,
                evaluated.pose.bones[name].matrix,
                reference.pose.bones[name].matrix_basis,
                expected.pose.bones[name].matrix,
                candidate.pose.bones[name].matrix_basis,
            ):
                _assert_finite_matrix(matrix, f"{name}, case {case}")
            for row in range(4):
                for column in range(4):
                    maximum_local = max(
                        maximum_local,
                        abs(
                            evaluated.pose.bones[name].matrix_basis[row][column]
                            - reference.pose.bones[name].matrix_basis[row][column]
                        ),
                    )
                    maximum_pose = max(
                        maximum_pose,
                        abs(
                            evaluated.pose.bones[name].matrix[row][column]
                            - expected.pose.bones[name].matrix[row][column]
                        ),
                    )
                    original_error = max(
                        original_error,
                        abs(
                            candidate.pose.bones[name].matrix_basis[row][column]
                            - original_basis[name][row * 4 + column]
                        ),
                    )
        passed = maximum_local <= 1e-5 and maximum_pose <= 1e-5 and original_error == 0.0
        results.append(
            {
                "case": case,
                "max_local_error": maximum_local,
                "max_pose_error": maximum_pose,
                "original_error": original_error,
                "passed": passed,
            }
        )
        print(
            f"BODY PARITY case={case} local={maximum_local:.9g} pose={maximum_pose:.9g} original={original_error:.9g}",
            flush=True,
        )
    report = {
        "scope": "native_body_matrix_parity",
        "binary": bpy.app.binary_path,
        "joint_count": len(joints),
        "cases": results,
        "passed": all(case["passed"] for case in results),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if not report["passed"]:
        raise AssertionError(f"Native body parity failed: {output}")
    print(f"NATIVE BODY PASS: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.fixture, args.output)
