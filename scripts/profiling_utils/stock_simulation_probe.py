"""Compare stock-Blender handler and driver ordering using a stateful GN consumer."""

from __future__ import annotations

import argparse
import json
import sys

from array import array
from pathlib import Path
from typing import Any

import bpy


sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_driver_benchmark import install_cached_carrier
from stock_riglogic_probe import make_rig, runtime_manifest, vertices


def make_accumulator(source: Any) -> Any:
    """Accumulate the source's deformed first vertex once per simulation frame."""
    mesh = bpy.data.meshes.new("SimulationSensor")
    mesh.from_pydata([(0, 0, 0)], [], [])
    sensor = bpy.data.objects.new("SimulationSensor", mesh)
    bpy.context.scene.collection.objects.link(sensor)
    group = bpy.data.node_groups.new("SimulationSensor", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    object_info = nodes.new("GeometryNodeObjectInfo")
    object_info.inputs["Object"].default_value = source
    position = nodes.new("GeometryNodeInputPosition")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(position.outputs["Position"], separate.inputs["Vector"])
    sample = nodes.new("GeometryNodeSampleIndex")
    sample.data_type = "FLOAT"
    sample.domain = "POINT"
    links.new(object_info.outputs["Geometry"], sample.inputs["Geometry"])
    links.new(separate.outputs["X"], sample.inputs["Value"])
    simulation_output = nodes.new("GeometryNodeSimulationOutput")
    simulation_output.state_items.new("FLOAT", "Total")
    simulation_input = nodes.new("GeometryNodeSimulationInput")
    simulation_input.pair_with_output(simulation_output)
    links.new(group_input.outputs["Geometry"], simulation_input.inputs["Geometry"])
    links.new(simulation_input.outputs["Geometry"], simulation_output.inputs["Geometry"])
    addition = nodes.new("ShaderNodeMath")
    addition.operation = "ADD"
    links.new(simulation_input.outputs["Total"], addition.inputs[0])
    links.new(sample.outputs["Value"], addition.inputs[1])
    links.new(addition.outputs[0], simulation_output.inputs["Total"])
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT"
    store.domain = "POINT"
    store.inputs["Name"].default_value = "accumulated_signal"
    links.new(simulation_output.outputs["Geometry"], store.inputs["Geometry"])
    links.new(simulation_output.outputs["Total"], store.inputs["Value"])
    links.new(store.outputs["Geometry"], group_output.inputs["Geometry"])
    modifier = sensor.modifiers.new("SimulationSensor", "NODES")
    modifier.node_group = group
    return sensor


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run one scheduling variant in a fresh process and persist every frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle", "handler", "driver", "cached-driver"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--bake", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    manifest = runtime_manifest()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, args.frames
    scene.frame_set(0)
    rig, mesh_object = make_rig()
    source = bpy.data.objects.new("AnimatedInput", None)
    scene.collection.objects.link(source)
    for frame in range(1, args.frames + 1):
        source.location.x = frame * 0.125
        source.keyframe_insert(data_path="location", index=0, frame=frame)
    source.location.x = 0
    sensor = make_accumulator(mesh_object)
    buffer = array("f", [0]) * (len(rig.pose.bones) * 3)
    events: list[dict[str, Any]] = []
    cache_state = None

    def apply_value(value: float) -> None:
        rig.pose.bones.foreach_get("location", buffer)
        buffer[0] = value
        rig.pose.bones.foreach_set("location", buffer)
        rig.update_tag(refresh={"OBJECT"})

    def post_frame(current_scene: Any, graph: Any) -> None:
        value = source.evaluated_get(graph).location.x
        apply_value(value)
        events.append({"frame": current_scene.frame_current, "input": value})

    def pre_frame(current_scene: Any, _graph: Any) -> None:
        apply_value(current_scene.frame_current * 0.125)

    if args.mode == "handler":
        bpy.app.handlers.frame_change_post.append(post_frame)
    elif args.mode == "oracle":
        bpy.app.handlers.frame_change_pre.append(pre_frame)
    elif args.mode in {"driver", "cached-driver"}:
        driver = rig.pose.bones[0].driver_add("location", 0).driver
        variable = driver.variables.new()
        variable.name = "signal"
        variable.type = "TRANSFORMS"
        variable.targets[0].id = source
        variable.targets[0].transform_type = "LOC_X"
        variable.targets[0].transform_space = "WORLD_SPACE"
        driver.expression = "signal"
        if args.mode == "cached-driver":
            carrier, cache_state = install_cached_carrier(source, True, input_path="location[0]", linear=True)
            variable.type = "SINGLE_PROP"
            variable.targets[0].id = carrier
            variable.targets[0].data_path = '["epoch"]'
            output = driver.variables.new()
            output.name = "cached"
            output.type = "SINGLE_PROP"
            output.targets[0].id = carrier
            output.targets[0].data_path = '["outputs"][0]'
            driver.expression = "cached + 0 * signal"
    frames = []
    cache_files = []
    try:
        bpy.context.view_layer.update()
        if args.bake:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cache_path = args.output.parent / f"cache-{args.mode}"
            if cache_path.exists():
                raise RuntimeError(f"Refusing to reuse an existing bake directory: {cache_path}")
            modifier = sensor.modifiers[0]
            modifier.bake_directory = str(cache_path)
            modifier.bake_target = "DISK"
            for bake in modifier.bakes:
                bake.use_custom_simulation_frame_range = True
                bake.frame_start, bake.frame_end = 1, args.frames
            bpy.ops.object.select_all(action="DESELECT")
            sensor.select_set(True)
            bpy.context.view_layer.objects.active = sensor
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix(".blend")), relative_remap=False)
            status = bpy.ops.object.simulation_nodes_cache_bake("EXEC_DEFAULT", selected=True)
            assert status == {"FINISHED"}, status
            cache_files = [str(path.relative_to(cache_path)) for path in cache_path.rglob("*") if path.is_file()]
            assert cache_files, "Bake produced no cache files"
        frame_order = range(args.frames, 0, -1) if args.bake else range(1, args.frames + 1)
        for frame in frame_order:
            scene.frame_set(frame)
            graph = bpy.context.evaluated_depsgraph_get()
            evaluated = sensor.evaluated_get(graph)
            attribute = evaluated.data.attributes.get("accumulated_signal")
            if attribute is None:
                raise AssertionError("Simulation produced no accumulator attribute")
            frames.append({"frame": frame, "mesh_x": vertices(mesh_object)[0][0], "total": attribute.data[0].value})
    finally:
        if args.mode == "handler":
            bpy.app.handlers.frame_change_post.remove(post_frame)
        elif args.mode == "oracle":
            bpy.app.handlers.frame_change_pre.remove(pre_frame)
    frames.sort(key=lambda item: item["frame"])
    result = {
        "runtime": manifest,
        "mode": args.mode,
        "baked": args.bake,
        "cache_files": cache_files,
        "frames": frames,
        "events": events,
    }
    assert all(abs(item["mesh_x"] - item["frame"] * 0.125) < 1e-6 for item in frames), frames
    assert frames[-1]["total"] > frames[0]["total"], "Simulation did not advance"
    if cache_state:
        result["cache_counts"] = {key: value for key, value in cache_state.items() if key != "values"}
        assert carrier["outputs"][0] == -1000.0
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        assert len(reference["frames"]) == len(frames)
        result["cache_errors"] = [
            abs(actual["total"] - expected["total"])
            for actual, expected in zip(frames, reference["frames"], strict=True)
        ]
        result["cache_parity"] = max(result["cache_errors"]) < 1e-6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.reference:
        assert result["cache_parity"], "Simulation cache differs from ordered oracle; see report"


if __name__ == "__main__":
    main()
