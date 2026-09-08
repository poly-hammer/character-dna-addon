"""Compare legacy and carrier rendering of the actual Ada scene in stock Blender."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys

from pathlib import Path

import bpy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling_utils.native_frame_benchmark import _load, snapshot
from profiling_utils.stock_riglogic_probe import runtime_manifest
from profiling_utils.stock_scene_binding import bind_scene


def main() -> None:  # noqa: PLR0915
    """Render both backends and ensure the render graph cannot alter viewport output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-trigger", action="store_true")
    parser.add_argument("--production", action="store_true")
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
    camera_data = bpy.data.cameras.new("StockParityCamera")
    camera = bpy.data.objects.new("StockParityCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, -3, 1.60)
    camera.rotation_euler.x = math.pi / 2
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 0.45
    scene.camera = camera
    light_data = bpy.data.lights.new("StockParityLight", "AREA")
    light = bpy.data.objects.new("StockParityLight", light_data)
    scene.collection.objects.link(light)
    light.location = (0, -1, 1.8)
    light.rotation_euler.x = math.pi / 2
    light_data.energy, light_data.size = 80.0, 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new("StockParityWorld")
    scene.world.color = (0.5, 0.5, 0.5)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.cycles.seed = 41
    scene.render.resolution_x = scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images = {"python": [], "carrier": []}
    viewport_errors = []
    binding = None
    for backend in ("python", "carrier"):
        if backend == "carrier":
            if args.production:
                from character_dna.runtime import controller, engine
                from character_dna.utilities import get_addon_preferences

                get_addon_preferences().experimental_native_riglogic = True
                bpy.ops.character_dna.sync_native_runtime()
                assert engine.active(instance), controller.status()
                binding = {"errors": []}
            else:
                binding = bind_scene(instance, native, args.array_trigger)
        for index, frame in enumerate((1, 57, 1)):
            scene.frame_set(frame)
            before = snapshot(instance, bpy.context.evaluated_depsgraph_get())
            path = args.output.with_name(f"{args.output.stem}-{backend}-{index}.exr")
            if path.exists():
                raise FileExistsError(path)
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(path), check_existing=False)
            pixels = np.asarray(image.pixels[:], dtype=np.float32)
            bpy.data.images.remove(image)
            assert pixels.size == 96 * 96 * 4 and np.isfinite(pixels).all()
            images[backend].append(pixels)
            after = snapshot(instance, bpy.context.evaluated_depsgraph_get())
            viewport_errors.append(
                max(float(np.max(np.abs(value - before[name]), initial=0)) for name, value in after.items())
            )
    pixel_errors = [
        float(np.max(np.abs(candidate - reference)))
        for candidate, reference in zip(images["carrier"], images["python"], strict=True)
    ]
    changed = float(np.max(np.abs(images["carrier"][0] - images["carrier"][1])))
    repeated = float(np.max(np.abs(images["carrier"][0] - images["carrier"][2])))
    result = {
        "runtime": manifest,
        "array_trigger": args.array_trigger,
        "production": args.production,
        "pixel_errors": pixel_errors,
        "viewport_errors": viewport_errors,
        "changed_frame_delta": changed,
        "repeated_frame_error": repeated,
        "callback_errors": binding["errors"],
        "scope": "Background Cycles render only; not UI F12 or cancellation",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    assert max(pixel_errors) < 1e-5 and max(viewport_errors) < 1e-5 and repeated < 1e-5 and changed > 0.001
    assert not binding["errors"]


if __name__ == "__main__":
    main()
