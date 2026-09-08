"""Validate actual native character geometry consumed by GN and cloth disk bakes."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import bpy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_simulation_probe import make_accumulator

from profiling_utils.native_frame_benchmark import _load, _remove_python_evaluation


def make_cloth(source: bpy.types.Object, source_index: int) -> bpy.types.Object:
    """Drive a pinned cloth's input geometry from an evaluated character vertex."""
    mesh = bpy.data.meshes.new("NativeCloth")
    mesh.from_pydata(
        [(column * 0.1, row * 0.1, 1.0) for row in range(3) for column in range(3)],
        [],
        [
            (row * 3 + col, row * 3 + col + 1, row * 3 + col + 4, row * 3 + col + 3)
            for row in range(2)
            for col in range(2)
        ],
    )
    cloth = bpy.data.objects.new("NativeCloth", mesh)
    bpy.context.scene.collection.objects.link(cloth)
    group = bpy.data.node_groups.new("CharacterClothInput", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    incoming = nodes.new("NodeGroupInput")
    outgoing = nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    info.inputs["Object"].default_value = source
    position = nodes.new("GeometryNodeInputPosition")
    sample = nodes.new("GeometryNodeSampleIndex")
    sample.data_type = "FLOAT_VECTOR"
    sample.domain = "POINT"
    sample.inputs["Index"].default_value = source_index
    links.new(info.outputs["Geometry"], sample.inputs["Geometry"])
    links.new(position.outputs["Position"], sample.inputs["Value"])
    move = nodes.new("GeometryNodeSetPosition")
    links.new(incoming.outputs["Geometry"], move.inputs["Geometry"])
    links.new(sample.outputs["Value"], move.inputs["Offset"])
    links.new(move.outputs["Geometry"], outgoing.inputs["Geometry"])
    modifier = cloth.modifiers.new("CharacterInput", "NODES")
    modifier.node_group = group
    pins = cloth.vertex_groups.new(name="Pins")
    pins.add([0, 1, 2], 1.0, "REPLACE")
    modifier = cloth.modifiers.new("Cloth", "CLOTH")
    modifier.settings.vertex_group_mass = "Pins"
    modifier.settings.quality = 5
    modifier.point_cache.frame_start, modifier.point_cache.frame_end = 1, 12
    modifier.point_cache.use_disk_cache = True
    return cloth


def main() -> None:  # noqa: PLR0915
    """Bake live native input and a test-only ordered oracle in fresh processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("oracle", "native"), required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    instance = _load(args.fixture)
    from character_dna import rig_instance
    from character_dna.runtime import controller, engine
    from character_dna.utilities import get_addon_preferences, get_addon_window_manager_properties

    get_addon_window_manager_properties().evaluate_dependency_graph = True
    rig_instance.start_listening()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 12
    points = []
    for frame in range(1, 13):
        scene.frame_set(frame)
        mesh = instance.head_mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        values = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", values)
        points.append(values.reshape((-1, 3)))
    index = int(np.argmax(np.ptp(np.stack(points)[:, :, 0], axis=0)))
    signal = [frame[index].tolist() for frame in points]
    assert max(value[0] for value in signal) - min(value[0] for value in signal) > 1e-4
    scene.frame_set(0)
    if args.mode == "native":
        get_addon_preferences().experimental_native_riglogic = True
        bpy.ops.character_dna.sync_native_runtime()
        assert engine.active(instance), controller.status()
        source = instance.head_mesh
    else:
        _remove_python_evaluation()
        mesh = bpy.data.meshes.new("OrderedOracle")
        mesh.from_pydata([signal[0]], [], [])
        source = bpy.data.objects.new("OrderedOracle", mesh)
        scene.collection.objects.link(source)

        def advance(current_scene: bpy.types.Scene, _graph: bpy.types.Depsgraph) -> None:
            source.data.vertices[0].co = signal[max(0, min(11, current_scene.frame_current - 1))]
            source.data.update()

        bpy.app.handlers.frame_change_pre.append(advance)
        index = 0
    sensor = make_accumulator(source)
    for node in sensor.modifiers[0].node_group.nodes:
        if node.bl_idname == "GeometryNodeSampleIndex":
            node.inputs["Index"].default_value = index
    cloth = make_cloth(source, index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    modifier = sensor.modifiers[0]
    cache = args.output.parent / f"cache-{args.mode}"
    if cache.exists():
        raise FileExistsError(cache)
    modifier.bake_directory = str(cache)
    modifier.bake_target = "DISK"
    for bake in modifier.bakes:
        bake.use_custom_simulation_frame_range = True
        bake.frame_start, bake.frame_end = 1, 12
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix(".blend")), relative_remap=False)
    bpy.ops.object.select_all(action="DESELECT")
    sensor.select_set(True)
    bpy.context.view_layer.objects.active = sensor
    assert bpy.ops.object.simulation_nodes_cache_bake("EXEC_DEFAULT", selected=True) == {"FINISHED"}
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    with bpy.context.temp_override(point_cache=cloth.modifiers[-1].point_cache):
        assert bpy.ops.ptcache.bake("EXEC_DEFAULT", bake=True) == {"FINISHED"}
    assert cloth.modifiers[-1].point_cache.is_baked
    frames = []
    for frame in range(12, 0, -1):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        mesh = cloth.evaluated_get(graph).data
        total = sensor.evaluated_get(graph).data.attributes["accumulated_signal"].data[0].value
        frames.append({"frame": frame, "total": total, "cloth": [list(vertex.co) for vertex in mesh.vertices]})
    result = {"mode": args.mode, "frames": frames, "oracle_signal": signal, "passed": True}
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        result["gn_error"] = max(
            abs(actual["total"] - expected["total"])
            for actual, expected in zip(frames, reference["frames"], strict=True)
        )
        result["cloth_error"] = max(
            float(np.max(np.abs(np.asarray(actual["cloth"]) - expected["cloth"])))
            for actual, expected in zip(frames, reference["frames"], strict=True)
        )
        assert result["gn_error"] < 1e-5 and result["cloth_error"] < 1e-5, result
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Live character bakes {args.mode}: GN={result.get('gn_error')} cloth={result.get('cloth_error')}")


if __name__ == "__main__":
    main()
