"""Compare real Ada evaluated scenes before and after disposable carrier binding."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time

from pathlib import Path
from typing import Any

import bpy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling_utils.native_frame_benchmark import _hash, _load, snapshot
from profiling_utils.stock_riglogic_probe import runtime_manifest
from profiling_utils.stock_scene_binding import bind_scene


def main() -> None:  # noqa: PLR0912, PLR0915
    """Fail the probe on graph errors or any evaluated output mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("parity", "frames"), default="parity")
    parser.add_argument("--backend", choices=("python", "carrier"), default="carrier")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--array-trigger", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--eye-aim", action="store_true")
    parser.add_argument("--profile-native", action="store_true")
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=120)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = runtime_manifest()
    instance = _load(args.fixture)
    if args.production:
        from character_dna.bindings import load_native_runtime

        native = load_native_runtime()
    else:
        if args.module_dir is None:
            raise ValueError("The prototype mode requires --module-dir")
        sys.path.insert(0, str(args.module_dir))
        native = importlib.import_module("_riglogic_blender")
    from character_dna import rig_instance
    from character_dna.utilities import get_addon_window_manager_properties

    get_addon_window_manager_properties().evaluate_dependency_graph = True
    rig_instance.start_listening()
    scene = bpy.context.scene
    if args.eye_aim:
        instance.face_board.pose.bones["CTRL_lookAtSwitch"].location.y = 1.0

    native_timings = {"head": [], "body": []}
    native_call = native.evaluate_frame if args.production else None
    if args.profile_native:
        if not args.production:
            raise ValueError("Native call profiling requires production mode")
        components = {}

        def timed_frame(*values: Any) -> int:
            session = values[0]
            if session not in components:
                components[session] = "head" if native.describe(session)["gui_names"] else "body"
            started = time.perf_counter_ns()
            result = native_call(*values)
            native_timings[components[session]].append((time.perf_counter_ns() - started) * 1e-6)
            return result

        native.evaluate_frame = timed_frame

    def bind_runtime() -> dict:
        if not args.production:
            return bind_scene(instance, native, args.array_trigger)
        from character_dna.runtime import controller, engine
        from character_dna.utilities import get_addon_preferences

        get_addon_preferences().experimental_native_riglogic = True
        bpy.ops.character_dna.sync_native_runtime()
        assert engine.active(instance), controller.status()
        return {"records": {}, "errors": [], "drivers": []}

    if args.stage == "frames":
        binding = bind_runtime() if args.backend == "carrier" else None
        for index in range(args.warmup):
            scene.frame_set(index % 120 + 1)
            bpy.context.evaluated_depsgraph_get()
        for timings in native_timings.values():
            timings.clear()
        samples = []
        for index in range(args.frames):
            started = time.perf_counter_ns()
            scene.frame_set((index + args.warmup) % 120 + 1)
            graph = bpy.context.evaluated_depsgraph_get()
            samples.append((time.perf_counter_ns() - started) * 1e-6)
        values = snapshot(instance, graph)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"), **values)
        ordered = sorted(samples)
        result = {
            "runtime": manifest,
            "backend": args.backend,
            "array_trigger": args.array_trigger,
            "production": args.production,
            "eye_aim": args.eye_aim,
            "frame_count": args.frames,
            "warmup": args.warmup,
            "fixture_sha256": _hash(args.fixture),
            "module_sha256": _hash(Path(native.__file__)),
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": ordered[int(len(ordered) * 0.95)],
            "p99_ms": ordered[int(len(ordered) * 0.99)],
            "samples_ms": samples,
        }
        if args.profile_native:
            native.evaluate_frame = native_call
            result["native_call_ms"] = {
                component: {"calls": len(values), "mean": statistics.mean(values)}
                for component, values in native_timings.items()
                if values
            }
        if binding:
            result["stages"] = {
                record["component"]: {
                    "calls": record["calls"],
                    "joint_bindings": len(record["plan"]),
                    "mean_ms": [value * 1000 / max(1, record["calls"]) for value in record["stage_seconds"]],
                }
                for record in binding["records"].values()
            }
            assert not binding["errors"] and all(driver.is_valid for driver in binding["drivers"])
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{args.backend}: mean={result['mean_ms']:.4f}ms p95={result['p95_ms']:.4f}ms; {args.output}")
        return
    frames = (*range(1, 121), 17, 57.25, 93.5, 1, 120) if args.trace else (1, 17, 57, 93, 120, 17)
    references = []
    for frame in frames:
        scene.frame_set(int(frame), subframe=frame % 1)
        references.append(snapshot(instance, bpy.context.evaluated_depsgraph_get()))
    binding = bind_runtime()
    results = []
    for frame, reference in zip(frames, references, strict=True):
        start = time.perf_counter()
        scene.frame_set(int(frame), subframe=frame % 1)
        graph = bpy.context.evaluated_depsgraph_get()
        elapsed = (time.perf_counter() - start) * 1000
        actual = snapshot(instance, graph)
        errors = {name: float(np.max(np.abs(values - reference[name]), initial=0)) for name, values in actual.items()}
        worst_bones = {}
        for component in ("head", "body"):
            bone_errors = np.max(np.abs(actual[f"{component}_pose"] - reference[f"{component}_pose"]), axis=1)
            rig = getattr(instance, f"{component}_rig")
            worst_bones[component] = sorted(
                [(bone.name, float(error)) for bone, error in zip(rig.pose.bones, bone_errors, strict=True)],
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        results.append(
            {
                "frame": frame,
                "elapsed_ms": elapsed,
                "errors": {name: value for name, value in errors.items() if value},
                "worst_bones": worst_bones,
            }
        )
    result = {
        "runtime": manifest,
        "array_trigger": args.array_trigger,
        "frames": results,
        "callback_errors": binding["errors"],
        "invalid_drivers": sum(not driver.is_valid for driver in binding["drivers"]),
        "stages": {
            record["component"]: {
                "calls": record["calls"],
                "joint_bindings": len(record["plan"]),
                "mean_ms": [value * 1000 / max(1, record["calls"]) for value in record["stage_seconds"]],
            }
            for record in binding["records"].values()
        },
        "scope": "Real scene topology probe, native mapping/solve/output math; Python input capture",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    maximum = max((error for frame in results for error in frame["errors"].values()), default=0.0)
    print(f"Scene parity: {len(results)} frames; max_error={maximum}; {args.output}")
    assert not binding["errors"] and not result["invalid_drivers"]
    assert max((error for frame in results for error in frame["errors"].values()), default=0.0) < 1e-5


if __name__ == "__main__":
    main()
