"""Reproducible Ada frame-evaluation comparison, excluding viewport drawing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time

from pathlib import Path
from typing import Any

import bpy
import numpy as np

from mathutils import Quaternion


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(fixture: Path) -> Any:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from profiling_utils.ci_benchmark import setup_environment

    bpy.ops.wm.open_mainfile(filepath=str(fixture))
    if not setup_environment():
        raise RuntimeError("Reference add-on unavailable")
    from character_dna.utilities import get_active_rig_instance

    instance = get_active_rig_instance()
    if instance is None:
        raise RuntimeError("No Ada instance in fixture")
    instance.initialize()
    return instance


def _remove_python_evaluation() -> None:
    from character_dna import rig_instance
    from character_dna.utilities import get_addon_window_manager_properties

    get_addon_window_manager_properties().evaluate_dependency_graph = False
    for name in dir(bpy.app.handlers):
        collection = getattr(bpy.app.handlers, name)
        if isinstance(collection, list):
            for handler in list(collection):
                if getattr(handler, "__module__", "") == rig_instance.__name__:
                    collection.remove(handler)
    if bpy.app.timers.is_registered(rig_instance.run_main_thread_evaluations):
        bpy.app.timers.unregister(rig_instance.run_main_thread_evaluations)
    for name in ("depsgraph_update_post", "frame_change_post"):
        if any(
            getattr(handler, "__module__", "") == rig_instance.__name__ for handler in getattr(bpy.app.handlers, name)
        ):
            raise AssertionError("Python RigLogic evaluation remains enabled")


def bind_native(instance: Any, diagnostic: bool = False) -> None:
    """Bind original scene targets once; subsequent evaluation is entirely native."""
    from profiling_utils.native_head_test import head_binding

    if not bpy.app.riglogic.supported:
        raise RuntimeError("WITH_RIGLOGIC is unavailable")
    _remove_python_evaluation()
    binding = head_binding(instance)
    binding["keys"] = [
        {
            "key": blocks.id_data,
            "positions": [int(value) for value in positions],
            "channels": [int(value) for value in channels],
        }
        for blocks, positions, channels, _driven, _buffer in instance.head_shape_key_apply_plan
    ]
    node = instance.head_texture_masks_node
    if node is None or not binding["keys"]:
        raise RuntimeError("Missing required native output targets")
    sockets = {socket.name: index for index, socket in enumerate(node.inputs)}
    binding["material"] = instance.head_material
    binding["texture_node"] = node.name
    binding["maps"] = [
        {"channel": channel, "socket": sockets[name]}
        for channel, name in instance.head_animated_map_plan
        if name in sockets
    ]
    binding["debug_eye"] = [0.0, 0.0, 0.0]
    instance.head_rig["_riglogic_head"] = binding
    instance.body_rig["_riglogic_body"] = {
        "path": bpy.path.abspath(instance.body_dna_file_path),
        "lod": 0,
        "drivers": [{"bone": name} for name in sorted(instance.body_driver_bone_names)],
        "joints": [
            {
                "index": index,
                "bone": name,
                "location": list(location),
                "rotation": list(rotation),
                "scale": list(scale),
                "rest_inverse": [value for row in inverse for value in row],
            }
            for index, name, location, rotation, scale, inverse in instance.body_bone_transform_plan
        ],
    }
    if diagnostic:
        instance.head_rig["_riglogic_head"]["debug_timings"] = [0.0] * 5
        instance.body_rig["_riglogic_body"]["debug_timings"] = [0.0] * 5
    bpy.app.riglogic.rebuild_bindings()
    bpy.context.view_layer.update()


def prepare(fixture: Path, output: Path) -> None:
    """Create a deterministic 120-frame action shared by both evaluation modes."""
    instance = _load(fixture)
    from character_dna.utilities import get_addon_window_manager_properties

    get_addon_window_manager_properties().evaluate_dependency_graph = False
    for obj in (instance.head_rig, instance.body_rig, instance.face_board):
        obj.animation_data_clear()
    instance.face_board.pose.bones["CTRL_lookAtSwitch"].location.y = 0.0
    for frame in range(1, 121):
        phase = (frame - 1) * 2.0 * math.pi / 120.0
        for index, name, axis in instance.head_gui_control_plan:
            bone = instance.face_board.pose.bones.get(name)
            if bone:
                setattr(bone.location, axis, math.sin(phase + index * 0.7) * 0.3)
                bone.keyframe_insert("location", index="xyz".index(axis), frame=frame)
        for index, name in enumerate(sorted(instance.body_driver_bone_names)):
            bone = instance.body_rig.pose.bones[name]
            bone.rotation_mode = "QUATERNION"
            axis = tuple(float(index % 3 == component) for component in range(3))
            bone.rotation_quaternion = Quaternion(axis, math.sin(phase + index * 0.3) * 0.15)
            bone.keyframe_insert("rotation_quaternion", frame=frame)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 120
    bpy.context.scene.frame_set(1)
    get_addon_window_manager_properties().evaluate_dependency_graph = True
    instance.evaluate()
    bpy.ops.wm.save_as_mainfile(filepath=str(output), relative_remap=False)
    print(f"ANIMATED FIXTURE: {output}; sha256={_hash(output)}", flush=True)


def snapshot(instance: Any, graph: bpy.types.Depsgraph) -> dict[str, np.ndarray]:
    """Read evaluated outputs without modifying or updating the graph."""
    arrays = {}
    for component in ("head", "body"):
        evaluated = getattr(instance, f"{component}_rig").evaluated_get(graph)
        arrays[f"{component}_pose"] = np.asarray(
            [[value for row in bone.matrix for value in row] for bone in evaluated.pose.bones], dtype=np.float32
        )
    for obj in sorted(bpy.data.objects, key=lambda obj: obj.name):
        if obj.type == "MESH":
            vertices = obj.evaluated_get(graph).data.vertices
            values = np.empty(len(vertices) * 3, dtype=np.float32)
            vertices.foreach_get("co", values)
            arrays[f"mesh_{obj.name}"] = values
    node = instance.head_material.evaluated_get(graph).node_tree.nodes[instance.head_texture_masks_node.name]
    arrays["masks"] = np.asarray(
        [node.inputs[name].default_value for _channel, name in instance.head_animated_map_plan if name in node.inputs],
        dtype=np.float32,
    )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise AssertionError("Non-finite evaluated benchmark output")
    return arrays


def benchmark(fixture: Path, output: Path, backend: str, warmup: int, frames: int, diagnostic: bool = False) -> None:
    """Measure synchronous animated scene evaluation; no viewport FPS is inferred."""
    instance = _load(fixture)
    if backend == "native":
        bind_native(instance, diagnostic)
    else:
        from character_dna import rig_instance
        from character_dna.utilities import get_addon_window_manager_properties

        get_addon_window_manager_properties().evaluate_dependency_graph = True
        rig_instance.start_listening()
        if rig_instance.frame_change_handler not in bpy.app.handlers.frame_change_post:
            raise AssertionError("Reference RigLogic frame handler is not registered")
        if "_riglogic_head" in instance.head_rig or "_riglogic_body" in instance.body_rig:
            raise AssertionError("Python baseline fixture contains native bindings")
    scene = bpy.context.scene
    for index in range(warmup):
        scene.frame_set(index % 120 + 1)
        bpy.context.evaluated_depsgraph_get()
    samples = []
    for index in range(frames):
        start = time.perf_counter_ns()
        scene.frame_set((index + warmup) % 120 + 1)
        graph = bpy.context.evaluated_depsgraph_get()
        samples.append(time.perf_counter_ns() - start)
    arrays = snapshot(instance, graph)
    snapshot_path = output.with_suffix(".npz")
    if snapshot_path.exists():
        raise FileExistsError(snapshot_path)
    np.savez_compressed(snapshot_path, **arrays)
    if backend == "native":
        diagnostics = instance.head_rig.evaluated_get(graph)["_riglogic_head"]["debug_eye"]
        if diagnostics[0] < 1 or diagnostics[2] != 1.0:
            raise AssertionError("Native head did not solve successfully")
    ordered = sorted(samples)
    report = {
        "schema_version": 1,
        "scope": "synchronous_scene_evaluation_no_viewport",
        "workload": "Ada LOD0 face GUI and body-driver-bone animation; no separate control rig",
        "backend": backend,
        "binary": bpy.app.binary_path,
        "binary_sha256": _hash(Path(bpy.app.binary_path)),
        "fixture": str(fixture),
        "fixture_sha256": _hash(fixture),
        "script_sha256": _hash(Path(__file__)),
        "frame_count": frames,
        "warmup": warmup,
        "last_frame": scene.frame_current,
        "times_ns": samples,
        "mean_ms": statistics.mean(samples) / 1e6,
        "median_ms": statistics.median(samples) / 1e6,
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] / 1e6,
        "p99_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] / 1e6,
        "evaluated_frames_per_second": frames * 1e9 / sum(samples),
        "viewport_fps": None,
        "output_snapshot": str(snapshot_path),
        "diagnostic": diagnostic,
    }
    if diagnostic and backend == "native":
        report["native_stages_including_warmup"] = {}
        for component in ("head", "body"):
            values = list(
                getattr(instance, f"{component}_rig").evaluated_get(graph)[f"_riglogic_{component}"]["debug_timings"]
            )
            if values[0] < frames:
                raise AssertionError(f"Native {component} solve count is too small: {values[0]}")
            report["native_stages_including_warmup"][component] = {
                "evaluations": values[0],
                "input_mean_ms": values[1] / values[0] / 1e6,
                "calculate_mean_ms": values[2] / values[0] / 1e6,
                "joint_apply_mean_ms": values[3] / values[0] / 1e6,
                "max_total_ms": values[4] / 1e6,
            }
        print(json.dumps(report["native_stages_including_warmup"], indent=2), flush=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        f"FRAME BENCHMARK {backend}: mean={report['mean_ms']:.4f}ms "
        f"p95={report['p95_ms']:.4f}ms eval_fps={report['evaluated_frames_per_second']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "frames"])
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["python", "native"], default="python")
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.frames < 1 or args.warmup < 0:
        raise ValueError("Frames must be positive and warmup nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args.fixture, args.output)
    else:
        benchmark(args.fixture, args.output, args.backend, args.warmup, args.frames, args.diagnostic)
