"""Exercise native editor value redraws and collapsed-panel expiry in Blender UI."""

# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path
from typing import Any

import bpy


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling_utils.native_frame_benchmark import _load


def main() -> None:  # noqa: PLR0915
    """Use actual UI panels and playback in a disposable Blender process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if bpy.app.background:
        raise RuntimeError("This probe requires Blender UI mode")
    if args.output.exists():
        raise FileExistsError(args.output)
    instance = _load(args.fixture)
    from character_dna.editors.raw_control_editor import callbacks as raw_callbacks
    from character_dna.editors.raw_control_editor.ui import CHARACTER_DNA_PT_raw_control_editor
    from character_dna.editors.shape_key_editor import callbacks as shape_callbacks
    from character_dna.editors.shape_key_editor.ui import CHARACTER_DNA_PT_shape_keys
    from character_dna.runtime import controller, engine, ui_refresh
    from character_dna.utilities import get_addon_preferences

    get_addon_preferences().experimental_native_riglogic = True
    bpy.ops.character_dna.sync_native_runtime()
    assert engine.active(instance), controller.status()
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.render.fps = 24
    scene.sync_mode = "NONE"
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    with bpy.context.temp_override(window=window, area=area):
        area.spaces.active.show_region_ui = True
    region = next(region for region in area.regions if region.type == "UI")
    window_region = next(region for region in area.regions if region.type == "WINDOW")
    raw_item = next(item for item in instance.raw_control_editor.raw_controls if item.name.endswith(".jawOpen"))
    shape_item = next(item for item in instance.shape_key_editor.shape_key_list if item.name.endswith("__brow_down_L"))
    draws: list[dict[str, Any]] = []
    state = {
        "stage": -1,
        "start": 0.0,
        "frame_events": 0,
        "legacy_reads": 0,
        "snapshot_reads": 0,
        "baseline_events": 0,
        "last_sample": -1,
        "deadline": time.monotonic() + 60,
    }
    measurements = []

    def frame_event(_scene: Any, _graph: Any) -> None:
        state["frame_events"] += 1

    bpy.app.handlers.frame_change_post.append(frame_event)
    original_snapshot = engine._module.control_snapshot
    original_legacy = engine.synchronize_legacy

    def snapshot(session: Any) -> Any:
        state["snapshot_reads"] += 1
        return original_snapshot(session)

    def legacy(*values: Any) -> Any:
        state["legacy_reads"] += 1
        return original_legacy(*values)

    engine._module.control_snapshot = snapshot
    engine.synchronize_legacy = legacy

    def raw_draw(panel: Any, context: Any) -> None:
        CHARACTER_DNA_PT_raw_control_editor.draw(panel, context)
        if state["stage"] == 1:
            draws.append(
                {
                    "kind": "raw",
                    "frame": scene.frame_current,
                    "value": raw_callbacks.get_raw_control_item_value(raw_item),
                }
            )

    def shape_draw(panel: Any, context: Any) -> None:
        CHARACTER_DNA_PT_shape_keys.draw(panel, context)
        if state["stage"] == 1:
            draws.append(
                {
                    "kind": "shape",
                    "frame": scene.frame_current,
                    "value": shape_callbacks.get_shape_key_value(shape_item),
                }
            )

    panels = []
    for category, closed in (("Native Live", False), ("Native Closed", True)):
        for kind, draw in (("Raw", raw_draw), ("Shapes", shape_draw)):
            name = f"NATIVE_TEST_PT_{kind}_{'closed' if closed else 'open'}"
            panel = type(
                name,
                (bpy.types.Panel,),
                {
                    "bl_idname": name,
                    "bl_label": kind,
                    "bl_space_type": "VIEW_3D",
                    "bl_region_type": "UI",
                    "bl_category": category,
                    "bl_options": {"DEFAULT_CLOSED"} if closed else set(),
                    "draw": draw,
                },
            )
            bpy.utils.register_class(panel)
            panels.append(panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stages = ("hidden-baseline", "open", "collapsed", "hidden-after")

    def finish(result: dict[str, Any]) -> None:
        engine._module.control_snapshot = original_snapshot
        engine.synchronize_legacy = original_legacy
        ui_refresh.clear()
        if window.screen.is_animation_playing:
            with bpy.context.temp_override(window=window, area=area, region=window_region):
                bpy.ops.screen.animation_cancel(restore_frame=False)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        bpy.ops.wm.quit_blender()

    def tick() -> float | None:
        try:
            now = time.monotonic()
            if now > state["deadline"]:
                raise RuntimeError("Panel playback probe timed out")
            if state["stage"] >= 0 and now - state["start"] < 5.0:
                if state["stage"] in (0, 2, 3) and now - state["start"] > 1.5:
                    assert not ui_refresh._subscribers, (stages[state["stage"]], ui_refresh._subscribers)
                    assert not bpy.app.timers.is_registered(ui_refresh._refresh)
                return 0.1
            if state["stage"] >= 0:
                measurements.append(
                    {
                        "stage": stages[state["stage"]],
                        "seconds": now - state["start"],
                        "frame_events": state["frame_events"] - state["baseline_events"],
                    }
                )
            state["stage"] += 1
            if state["stage"] == len(stages):
                values = {kind: [item for item in draws if item["kind"] == kind] for kind in ("raw", "shape")}
                for kind, samples in values.items():
                    assert len({item["frame"] for item in samples}) >= 5, (kind, samples)
                    assert max(item["value"] for item in samples) - min(item["value"] for item in samples) > 0.01, kind
                assert state["legacy_reads"] == 0, state
                finish(
                    {
                        "passed": True,
                        "measurements": measurements,
                        "draws": draws,
                        "legacy_reads": state["legacy_reads"],
                        "native_snapshot_reads": state["snapshot_reads"],
                        "refresh_hz_limit": 1 / ui_refresh.PLAYBACK_INTERVAL,
                        "collapsed_and_hidden_timer_stopped": True,
                    }
                )
                return None
            state["start"] = now
            state["baseline_events"] = state["frame_events"]
            with bpy.context.temp_override(window=window, area=area):
                area.spaces.active.show_region_ui = state["stage"] in (1, 2)
                if state["stage"] in (1, 2):
                    region.active_panel_category = "Native Live" if state["stage"] == 1 else "Native Closed"
                    region.tag_redraw()
            if not window.screen.is_animation_playing:
                with bpy.context.temp_override(window=window, area=area, region=window_region):
                    bpy.ops.screen.animation_play()
            return 0.1
        except Exception as error:
            finish({"passed": False, "error": str(error), "measurements": measurements, "draws": draws})
            return None

    bpy.app.timers.register(tick, first_interval=1.0)


if __name__ == "__main__":
    main()
