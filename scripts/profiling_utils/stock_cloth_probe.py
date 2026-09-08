"""Test live armature input ordering during actual cloth point-cache baking."""

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
from stock_riglogic_probe import make_rig, runtime_manifest


def main() -> None:  # noqa: PLR0915
    """Bake a partly pinned cloth, then inspect its cache backwards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle", "handler", "driver", "cached-driver"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    manifest = runtime_manifest()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 12
    scene.frame_set(0)
    rig, _unused_mesh = make_rig()
    source = bpy.data.objects.new("ClothInput", None)
    scene.collection.objects.link(source)
    for frame in range(1, 13):
        source.location.x = frame * 0.125
        source.keyframe_insert(data_path="location", index=0, frame=frame)
    source.location.x = 0
    mesh = bpy.data.meshes.new("ClothProbe")
    points = [(column * 0.25, row * 0.25, 1.0) for row in range(3) for column in range(3)]
    faces = [
        (row * 3 + column, row * 3 + column + 1, row * 3 + column + 4, row * 3 + column + 3)
        for row in range(2)
        for column in range(2)
    ]
    mesh.from_pydata(points, [], faces)
    cloth_object = bpy.data.objects.new("ClothProbe", mesh)
    scene.collection.objects.link(cloth_object)
    group = cloth_object.vertex_groups.new(name="Driven")
    group.add(list(range(9)), 1.0, "REPLACE")
    pins = cloth_object.vertex_groups.new(name="Pins")
    pins.add([0, 1, 2], 1.0, "REPLACE")
    deform = cloth_object.modifiers.new("Armature", "ARMATURE")
    deform.object = rig
    cloth = cloth_object.modifiers.new("Cloth", "CLOTH")
    cloth.settings.vertex_group_mass = pins.name
    cloth.settings.quality = 5
    cloth.point_cache.frame_start, cloth.point_cache.frame_end = 1, 12
    cloth.point_cache.use_disk_cache = True
    buffer = array("f", [0]) * (len(rig.pose.bones) * 3)
    events = []
    cache_state = None

    def set_input(value: float) -> None:
        rig.pose.bones.foreach_get("location", buffer)
        buffer[0] = value
        rig.pose.bones.foreach_set("location", buffer)
        rig.update_tag(refresh={"OBJECT"})

    def before(current_scene: Any, _graph: Any) -> None:
        set_input(current_scene.frame_current * 0.125)

    def after(current_scene: Any, graph: Any) -> None:
        value = source.evaluated_get(graph).location.x
        set_input(value)
        events.append({"frame": current_scene.frame_current, "value": value})

    if args.mode == "oracle":
        bpy.app.handlers.frame_change_pre.append(before)
    elif args.mode == "handler":
        bpy.app.handlers.frame_change_post.append(after)
    else:
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
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        blend_path = args.output.with_suffix(".blend")
        if blend_path.exists():
            raise RuntimeError(f"Refusing to reuse existing fixture: {blend_path}")
        bpy.ops.object.select_all(action="DESELECT")
        cloth_object.select_set(True)
        bpy.context.view_layer.objects.active = cloth_object
        bpy.context.view_layer.update()
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), relative_remap=False)
        with bpy.context.temp_override(point_cache=cloth.point_cache):
            status = bpy.ops.ptcache.bake("EXEC_DEFAULT", bake=True)
        assert status == {"FINISHED"} and cloth.point_cache.is_baked, status
        for frame in range(12, 0, -1):
            scene.frame_set(frame)
            evaluated = cloth_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
            frames.append({"frame": frame, "vertices": [list(vertex.co) for vertex in evaluated.data.vertices]})
    finally:
        if args.mode == "oracle":
            bpy.app.handlers.frame_change_pre.remove(before)
        elif args.mode == "handler":
            bpy.app.handlers.frame_change_post.remove(after)
    frames.sort(key=lambda item: item["frame"])
    assert frames[-1]["vertices"] != frames[0]["vertices"], "Cloth did not advance"
    result = {"runtime": manifest, "mode": args.mode, "baked": True, "frames": frames, "events": events}
    if cache_state:
        result["cache_counts"] = {key: value for key, value in cache_state.items() if key != "values"}
        assert carrier["outputs"][0] == -1000.0
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        errors = [
            max(
                abs(value - expected_value)
                for vertex, expected_vertex in zip(actual["vertices"], expected["vertices"], strict=True)
                for value, expected_value in zip(vertex, expected_vertex, strict=True)
            )
            for actual, expected in zip(frames, reference["frames"], strict=True)
        ]
        result.update(cache_errors=errors, cache_parity=max(errors) < 1e-5)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Cloth bake {args.mode}: cache_parity={result.get('cache_parity', 'oracle')}; report: {args.output}")
    if args.reference:
        assert result["cache_parity"], "Cloth cache differs from ordered oracle; see report"


if __name__ == "__main__":
    main()
