"""Measure a diagnostic floor for scalar driver dispatch, not RigLogic speed."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time

from array import array
from pathlib import Path
from typing import Any

import bpy


sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_benchmark import ADDON_ROOT, make_raw_inputs
from stock_riglogic_probe import runtime_manifest


def install_cached_carrier(
    source: bpy.types.Object,
    buffer_output: bool = False,
    output_count: int = 1,
    input_path: str = '["signal"]',
    linear: bool = False,
    native: Any = None,
) -> tuple[bpy.types.Object, dict]:
    """Test public RNA reads of a single dependency-ordered computation cache."""
    carrier = bpy.data.objects.new("CachedDriverOutput", None)
    bpy.context.scene.collection.objects.link(carrier)
    carrier["epoch"] = 0.0
    native_sessions = {}
    if native:
        model = native.load_model(str(ADDON_ROOT / "tests/test_files/dna/ada/head.dna"))
        description = native.describe(native.create_session(model))
        output_count = sum(description["output_counts"])
        base = array("f", make_raw_inputs(description["raw_names"], 0))
        control_index = description["raw_names"].index("CTRL_expressions.jawOpen")
    carrier["outputs"] = array("d", [-1000.0]) * output_count
    state = {"values": {}, "calls": 0, "solves": 0, "reads": 0, "misses": 0}

    def solve(owner: bpy.types.Object, signal: float) -> float:
        state["calls"] += 1
        key = owner.as_pointer()
        if native:
            if not owner.is_evaluated or owner == carrier:
                raise RuntimeError("Driver must publish only to its evaluated scratch carrier")
            if key not in native_sessions:
                native_sessions[key] = (native.create_session(model), array("f", base))
            session, controls = native_sessions[key]
            controls[control_index] = signal
            native.evaluate_into(session, controls, owner["outputs"])
            state["solves"] = sum(native.session_statistics(value[0])["solves"] for value in native_sessions.values())
            return signal
        previous = state["values"].get(key)
        if previous is None or previous[0] != signal:
            state["solves"] += 1
            output_value = signal if linear else math.sin(signal)
            state["values"][key] = (
                signal,
                array("d", (output_value + index * 1e-6 for index in range(output_count))),
            )
        if buffer_output:
            if not owner.is_evaluated or owner == carrier:
                raise RuntimeError("Driver must publish only to its evaluated scratch carrier")
            memoryview(owner["outputs"])[:] = memoryview(state["values"][key][1])
        return signal

    def read(owner: bpy.types.Object) -> float:
        state["reads"] += 1
        value = state["values"].get(owner.as_pointer())
        if value is None:
            state["misses"] += 1
            return -1000.0
        return value[1][0]

    bpy.types.Object.rl_probe_cached = bpy.props.FloatProperty(get=read)
    bpy.app.driver_namespace["rl_probe_solve"] = solve
    driver = carrier.driver_add('["epoch"]').driver
    driver.use_self = True
    variable = driver.variables.new()
    variable.name = "signal"
    variable.type = "SINGLE_PROP"
    variable.targets[0].id = source
    variable.targets[0].data_path = input_path
    driver.expression = "rl_probe_solve(self, signal)"
    return carrier, state


def main() -> None:  # noqa: PLR0915
    """Compare Blender's simple-expression path with a registered C callable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("simple", "native-call", "rna-cache", "buffer-cache", "native-buffer"), required=True
    )
    parser.add_argument("--module-dir", type=Path)
    parser.add_argument("--bones", type=int, default=1118)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    manifest = runtime_manifest()
    scene = bpy.context.scene
    scene.frame_set(0)
    source = bpy.data.objects.new("DriverInput", None)
    scene.collection.objects.link(source)
    for frame, value in ((1, 0.1), (args.frames + args.warmup + 1, 0.4)):
        source["signal"] = value
        source.keyframe_insert(data_path='["signal"]', frame=frame)
    armature = bpy.data.armatures.new("DriverDispatch")
    rig = bpy.data.objects.new("DriverDispatch", armature)
    scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for index in range(args.bones):
        bone = armature.edit_bones.new(f"Driven{index:04d}")
        bone.head, bone.tail = (index * 0.01, 0, 0), (index * 0.01, 0, 0.01)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.app.driver_namespace["rl_probe"] = math.sin
    cache_state = None
    native = None
    if args.backend == "native-buffer":
        if args.module_dir is None:
            raise ValueError("Native buffer mode requires --module-dir")
        sys.path.insert(0, str(args.module_dir))
        native = importlib.import_module("_riglogic_blender")
    if args.backend in {"rna-cache", "buffer-cache", "native-buffer"}:
        carrier, cache_state = install_cached_carrier(
            source, args.backend != "rna-cache", args.bones * 9 if args.backend == "buffer-cache" else 1, native=native
        )
        if native and args.bones * 9 > len(carrier["outputs"]):
            raise ValueError("Requested more output channels than the native head model produces")
    drivers = []
    for bone in rig.pose.bones:
        bone.rotation_mode = "XYZ"
        for property_name in ("location", "rotation_euler", "scale"):
            for axis in range(3):
                curve = bone.driver_add(property_name, axis)
                curve.keyframe_points.clear()
                for modifier in tuple(curve.modifiers):
                    curve.modifiers.remove(modifier)
                driver = curve.driver
                variable = driver.variables.new()
                variable.name = "signal"
                variable.type = "SINGLE_PROP"
                variable.targets[0].id = source
                variable.targets[0].data_path = '["signal"]'
                expression = "sin(signal)" if args.backend == "simple" else "rl_probe(signal)"
                if args.backend in {"rna-cache", "buffer-cache", "native-buffer"}:
                    variable.targets[0].id = carrier
                    variable.targets[0].data_path = '["epoch"]'
                    output = driver.variables.new()
                    output.name = "cached"
                    output.type = "SINGLE_PROP"
                    output.targets[0].id = carrier
                    output.targets[0].data_path = (
                        "rl_probe_cached" if args.backend == "rna-cache" else f'["outputs"][{len(drivers)}]'
                    )
                    expression = "cached + 0 * signal"
                driver.expression = expression + ("+1" if property_name == "scale" else "")
                drivers.append(driver)
    bpy.context.view_layer.update()
    assert all(driver.is_valid for driver in drivers)
    assert all(driver.is_simple_expression == (args.backend != "native-call") for driver in drivers)
    samples = []
    for frame in range(1, args.warmup + args.frames + 1):
        start = time.perf_counter_ns()
        scene.frame_set(frame)
        elapsed = (time.perf_counter_ns() - start) * 1e-6
        if frame > args.warmup:
            samples.append(elapsed)
    graph = bpy.context.evaluated_depsgraph_get()
    expected = math.sin(source.evaluated_get(graph)["signal"])
    expected_outputs = None
    if native:
        reference = native.create_session(native.load_model(str(ADDON_ROOT / "tests/test_files/dna/ada/head.dna")))
        description = native.describe(reference)
        controls = array("f", make_raw_inputs(description["raw_names"], 0))
        controls[description["raw_names"].index("CTRL_expressions.jawOpen")] = source.evaluated_get(graph)["signal"]
        values = native.evaluate(reference, controls)
        expected_outputs = [value for family in ("joints", "blend_shapes", "animated_maps") for value in values[family]]
    evaluated = rig.evaluated_get(graph)
    maximum_error = max(
        abs(
            value
            - (
                (expected_outputs[bone_index * 9 + property_index * 3 + axis] if expected_outputs else expected)
                + (property_name == "scale")
                + ((bone_index * 9 + property_index * 3 + axis) * 1e-6 if args.backend == "buffer-cache" else 0)
            )
        )
        for bone_index, bone in enumerate(evaluated.pose.bones)
        for property_index, property_name in enumerate(("location", "rotation_euler", "scale"))
        for axis, value in enumerate(getattr(bone, property_name))
    )
    assert maximum_error < 1e-6, maximum_error
    ordered = sorted(samples)
    result = {
        "runtime": manifest,
        "backend": args.backend,
        "bone_count": args.bones,
        "scalar_drivers": len(drivers),
        "frames": args.frames,
        "warmup": args.warmup,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int((len(ordered) - 1) * 0.95)],
        "max_output_error": maximum_error,
        "samples_ms": samples,
        "cache_counts": {key: value for key, value in cache_state.items() if key != "values"} if cache_state else None,
        "scope": "SDK plus identity bone-driver dispatch" if native else "Synthetic driver-dispatch floor",
        "exclusions": "No rest-space conversion, mesh deformation, material application or complete scene parity",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.backend}: {len(drivers)} drivers, {result['mean_ms']:.4f} ms; report: {args.output}")
    if cache_state:
        assert cache_state["misses"] == 0, cache_state
        assert carrier["outputs"][0] == -1000.0, "Evaluated output write leaked into original datablock"


if __name__ == "__main__":
    main()
