"""Run actual modal UI render jobs against the packaged native backend."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

from pathlib import Path
from typing import Any

import bpy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling_utils.native_frame_benchmark import _load, snapshot


def main() -> None:  # noqa: PLR0915
    """Use a main-thread timer for UI job coordination; never block render callbacks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if bpy.app.background:
        raise RuntimeError("This test requires Blender UI mode")
    instance = _load(args.fixture)
    from character_dna.runtime import controller, engine
    from character_dna.utilities import get_addon_preferences

    get_addon_preferences().experimental_native_riglogic = True
    bpy.ops.character_dna.sync_native_runtime()
    assert engine.active(instance), controller.status()
    scene = bpy.context.scene
    data = bpy.data.cameras.new("RuntimeUICamera")
    camera = bpy.data.objects.new("RuntimeUICamera", data)
    scene.collection.objects.link(camera)
    camera.location = (0, -3, 1.6)
    camera.rotation_euler.x = math.pi / 2
    data.type, data.ortho_scale = "ORTHO", 0.45
    scene.camera = camera
    data = bpy.data.lights.new("RuntimeUILight", "AREA")
    light = bpy.data.objects.new("RuntimeUILight", data)
    scene.collection.objects.link(light)
    light.location = (0, -1, 1.8)
    light.rotation_euler.x = math.pi / 2
    data.energy, data.size = 80, 1
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.cycles.seed = 41
    scene.render.resolution_x = scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "index": 0,
        "running": False,
        "finished": False,
        "cancelled": False,
        "cancel_sent": False,
        "deadline": time.monotonic() + 120,
    }
    images = []
    errors = []

    def finished(*_args: Any) -> None:
        state["finished"] = True

    bpy.app.handlers.render_complete.append(finished)

    def cancelled(*_args: Any) -> None:
        state["cancelled"] = True
        state["finished"] = True

    bpy.app.handlers.render_cancel.append(cancelled)

    def tick() -> float | None:
        try:
            if time.monotonic() > state["deadline"]:
                raise RuntimeError("UI render test timed out")
            if state["running"]:
                if (
                    args.cancel
                    and state["index"] == 2
                    and bpy.app.is_job_running("RENDER")
                    and time.monotonic() - state["job_started"] > 1.0
                ):
                    for window in bpy.context.window_manager.windows:
                        window.event_simulate(type="ESC", value="PRESS")
                        window.event_simulate(type="ESC", value="RELEASE")
                    state["cancel_sent"] = True
                if not state["finished"] or bpy.app.is_job_running("RENDER"):
                    return 0.05
                after = snapshot(instance, bpy.context.evaluated_depsgraph_get())
                error = max(
                    float(np.max(np.abs(value - state["before"][name]), initial=0)) for name, value in after.items()
                )
                assert error < 1e-5, error
                errors.append(error)
                if args.cancel and state["index"] == 2:
                    assert state["cancelled"], "The requested render did not cancel"
                else:
                    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
                    pixels = np.asarray(image.pixels[:], dtype=np.float32)
                    bpy.data.images.remove(image)
                    assert np.isfinite(pixels).all() and pixels.size == 96 * 96 * 4
                    images.append(pixels)
                state["index"] += 1
                state["running"] = False
            if state["index"] == (4 if args.cancel else 2):
                change = float(np.max(np.abs(images[0] - images[1])))
                assert change > 0.001, change
                restart_error = float(np.max(np.abs(images[0] - images[-1]))) if args.cancel else None
                if args.cancel:
                    assert restart_error < 1e-5, restart_error
                result = {
                    "passed": True,
                    "ui_render_jobs": state["index"],
                    "viewport_errors": errors,
                    "pixel_change": change,
                    "cancelled": state["cancelled"],
                    "restart_error": restart_error,
                }
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(result), flush=True)
                bpy.ops.wm.quit_blender()
                return None
            frame = (1, 57, 17, 1)[state["index"]]
            scene.cycles.samples = 8192 if args.cancel and state["index"] == 2 else 4
            scene.render.resolution_x = scene.render.resolution_y = 2048 if args.cancel and state["index"] == 2 else 96
            scene.frame_set(frame)
            state["before"] = snapshot(instance, bpy.context.evaluated_depsgraph_get())
            scene.render.filepath = str(args.output.with_name(f"ui-{frame}.exr"))
            state["running"], state["finished"] = True, False
            state["job_started"] = time.monotonic()
            bpy.ops.render.render("INVOKE_DEFAULT", write_still=True)
            return 0.05
        except Exception as error:
            args.output.write_text(json.dumps({"passed": False, "error": str(error)}) + "\n", encoding="utf-8")
            print(f"UI RENDER FAILURE: {error}", flush=True)
            bpy.ops.wm.quit_blender()
            return None

    bpy.app.timers.register(tick, first_interval=0.5)


if __name__ == "__main__":
    main()
