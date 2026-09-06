"""Verify native-only render graph updates and deterministic frame revisits."""

from __future__ import annotations

import argparse
import json
import math
import sys

from pathlib import Path

import bpy
import numpy as np


def run(args: argparse.Namespace) -> None:
    from profiling_utils.native_isolation_test import pose

    bpy.ops.wm.open_mainfile(filepath=str(args.fixture))
    if "character_dna" in bpy.context.preferences.addons:
        raise AssertionError("Render test must not enable Python RigLogic")
    head = next(obj for obj in bpy.data.objects if "_riglogic_head" in obj)
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("NativeTestCamera")
    camera = bpy.data.objects.new("NativeTestCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, -3.0, 1.60)
    camera.rotation_euler.x = math.pi / 2.0
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 0.45
    scene.camera = camera
    if scene.world is None:
        scene.world = bpy.data.worlds.new("NativeTestWorld")
    scene.world.color = (0.5, 0.5, 0.5)
    light_data = bpy.data.lights.new("NativeTestLight", "AREA")
    light = bpy.data.objects.new("NativeTestLight", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, -1.0, 1.8)
    light.rotation_euler.x = math.pi / 2.0
    light_data.energy = 80.0
    light_data.size = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.cycles.seed = 41
    scene.render.resolution_x = scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    images = []
    for index, frame in enumerate((1, 57, 1)):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        before = pose(head, graph)
        path = args.output.with_name(f"{args.output.stem}-{index}.exr")
        if path.exists():
            raise FileExistsError(path)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = np.asarray(image.pixels[:], dtype=np.float32)
        bpy.data.images.remove(image)
        if not np.isfinite(pixels).all() or pixels.size != 96 * 96 * 4:
            raise AssertionError("Invalid rendered image")
        images.append(pixels)
        after = pose(head, bpy.context.evaluated_depsgraph_get())
        if not np.allclose(before, after, rtol=0, atol=1e-5):
            raise AssertionError("Render changed the viewport graph's evaluated pose")
    change = float(np.max(np.abs(images[0] - images[1])))
    repeated_error = float(np.max(np.abs(images[0] - images[2])))
    if change < 0.001 or repeated_error > 1e-5:
        raise AssertionError(f"Render did not update deterministically: change={change}, repeat={repeated_error}")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "passed": True,
                "changed_frame_pixel_delta": change,
                "repeated_frame_pixel_error": repeated_error,
                "python_addon_enabled": False,
            },
            stream,
            indent=2,
        )
    print(f"NATIVE RENDER PASS: changed={change:.6g}, repeat={repeated_error:.6g}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
