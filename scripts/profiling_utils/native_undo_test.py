"""Test native runtime reconstruction through Blender's undo and redo stack."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import bpy
import numpy as np


class NativeControlEdit(bpy.types.Operator):
    bl_idname = "wm.native_riglogic_control_test"
    bl_label = "Native RigLogic Control Test"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        head = next(obj for obj in context.scene.objects if "_riglogic_head" in obj)
        face = head["_riglogic_head"]["face_board"]
        face.animation_data_clear()
        face.pose.bones["CTRL_C_jaw"].location.y = 0.8
        return {"FINISHED"}


def run(args: argparse.Namespace) -> None:
    from profiling_utils.native_isolation_test import pose

    bpy.ops.wm.open_mainfile(filepath=str(args.fixture))
    bpy.context.preferences.edit.use_global_undo = True
    bpy.utils.register_class(NativeControlEdit)
    bpy.context.scene.frame_set(47)
    head_name = next(obj.name for obj in bpy.data.objects if "_riglogic_head" in obj)
    graph = bpy.context.evaluated_depsgraph_get()
    baseline = pose(bpy.data.objects[head_name], graph)
    bpy.ops.ed.undo_push(message="Native Baseline")
    bpy.ops.wm.native_riglogic_control_test()
    bpy.context.view_layer.update()
    changed = pose(bpy.data.objects[head_name], bpy.context.evaluated_depsgraph_get())
    if np.allclose(baseline, changed, rtol=0, atol=1e-5):
        raise AssertionError("Undo test did not change native output")
    bpy.ops.ed.undo_push(message="Native Changed")
    bpy.ops.ed.undo()
    bpy.context.view_layer.update()
    restored = pose(bpy.data.objects[head_name], bpy.context.evaluated_depsgraph_get())
    if not np.allclose(baseline, restored, rtol=0, atol=1e-5):
        raise AssertionError("Undo did not restore native output")
    bpy.ops.ed.redo()
    bpy.context.view_layer.update()
    redone = pose(bpy.data.objects[head_name], bpy.context.evaluated_depsgraph_get())
    if not np.allclose(changed, redone, rtol=0, atol=1e-5):
        raise AssertionError("Redo did not restore native output")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "passed": True,
                "undo_max_error": float(np.max(np.abs(baseline - restored))),
                "redo_max_error": float(np.max(np.abs(changed - redone))),
            },
            stream,
            indent=2,
        )
    print("NATIVE UNDO/REDO PASS", flush=True)


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
