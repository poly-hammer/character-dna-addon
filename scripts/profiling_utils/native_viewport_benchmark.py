"""Measure interactive viewport draw throughput during Blender-native playback."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

from pathlib import Path

import bpy

from mathutils import Quaternion, Vector


def run(args: argparse.Namespace) -> None:
    from profiling_utils.native_frame_benchmark import _hash, _load, bind_native

    if bpy.app.background:
        raise RuntimeError("Viewport benchmark must run with a window")
    instance = _load(args.fixture)
    if args.backend == "native":
        bind_native(instance)
    else:
        from character_dna import rig_instance
        from character_dna.utilities import get_addon_window_manager_properties

        get_addon_window_manager_properties().evaluate_dependency_graph = True
        rig_instance.start_listening()
    window = bpy.context.window or next(iter(bpy.context.window_manager.windows), None)
    if window is None:
        raise RuntimeError("No Blender window is available for viewport capture")
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    space = area.spaces.active
    region = next(region for region in area.regions if region.type == "WINDOW")
    space.overlay.show_overlays = False
    space.shading.type = args.shading
    space.region_3d.view_location = Vector((0.0, 0.0, 1.0))
    space.region_3d.view_distance = 3.1
    space.region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0), math.pi / 2.0)
    space.region_3d.view_perspective = "ORTHO"
    scene = bpy.context.scene
    scene.sync_mode = "NONE"
    scene.render.fps = 1000
    scene.render.fps_base = 1.0
    scene.frame_set(1)
    observations = []
    state = {"started": None, "handle": None}

    def observe() -> None:
        if bpy.context.area == area and state["started"] is not None:
            timestamp = time.perf_counter_ns()
            if timestamp >= state["started"] + int(args.warmup_seconds * 1e9):
                observations.append((timestamp, scene.frame_current))

    def start() -> None:
        state["handle"] = bpy.types.SpaceView3D.draw_handler_add(observe, (), "WINDOW", "POST_PIXEL")
        state["started"] = time.perf_counter_ns()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.screen.animation_play()
        bpy.app.timers.register(finish, first_interval=args.warmup_seconds + args.seconds)

    def finish() -> None:
        with bpy.context.temp_override(window=window, area=area, region=region):
            if window.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
        bpy.types.SpaceView3D.draw_handler_remove(state["handle"], "WINDOW")
        samples = [
            (timestamp, frame)
            for index, (timestamp, frame) in enumerate(observations)
            if index == 0 or frame != observations[index - 1][1]
        ]
        passed = len(samples) >= 3 and samples[-1][0] > samples[0][0]
        elapsed = (samples[-1][0] - samples[0][0]) / 1e9 if passed else 0.0
        screenshot = args.output.with_suffix(".png")
        bpy.ops.screen.screenshot(filepath=str(screenshot))
        report = {
            "scope": "viewport_post_pixel_observer_not_presentmon",
            "backend": args.backend,
            "shading": args.shading,
            "fixture_sha256": _hash(args.fixture),
            "binary_sha256": _hash(Path(bpy.app.binary_path)),
            "window_pixels": [window.width, window.height],
            "region_pixels": [region.width, region.height],
            "warmup_seconds": args.warmup_seconds,
            "requested_seconds": args.seconds,
            "playback_fps_target": 1000,
            "sync_mode": scene.sync_mode,
            "distinct_drawn_frames": len(samples),
            "elapsed_observer_seconds": elapsed,
            "drawn_animation_frames_per_second": (len(samples) - 1) / elapsed if passed else None,
            "presented_fps": None,
            "observations_ns_frame": observations,
            "screenshot": str(screenshot),
            "passed": passed,
        }
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(
            f"VIEWPORT {args.backend}/{args.shading}: {report['drawn_animation_frames_per_second']} drawn frames/s",
            flush=True,
        )
        bpy.ops.wm.quit_blender()

    bpy.app.timers.register(start, first_interval=1.0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["native", "python"], required=True)
    parser.add_argument("--shading", choices=["SOLID", "MATERIAL"], default="MATERIAL")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError(args.output)
    if args.seconds <= 0 or args.warmup_seconds < 0:
        raise ValueError("Invalid capture duration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
