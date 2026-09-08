"""Exercise the packaged native backend through its real preference and operator."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import bpy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling_utils.native_frame_benchmark import _load, snapshot


def main() -> None:  # noqa: PLR0915
    """Compare production binding output and toggle/load behavior against legacy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--undo", action="store_true")
    parser.add_argument("--duplicate", action="store_true")
    parser.add_argument("--editor", action="store_true")
    parser.add_argument("--inherit-scale", choices=("FULL", "FIX_SHEAR", "ALIGNED", "AVERAGE", "NONE", "NONE_LEGACY"))
    parser.add_argument("--no-inherit-rotation", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    instance = _load(args.fixture)
    if args.inherit_scale:
        bone = instance.head_rig.data.bones["FACIAL_C_FacialRoot"]
        bone.inherit_scale = args.inherit_scale
        bone.use_inherit_rotation = not args.no_inherit_rotation
        bone.use_local_location = False
        instance.head_rig.scale = (1.1, 0.9, 1.2)
    from character_dna import rig_instance
    from character_dna.runtime import controller, engine
    from character_dna.utilities import (
        get_active_rig_instance,
        get_addon_preferences,
        get_addon_window_manager_properties,
    )

    preferences = get_addon_preferences()
    preferences.experimental_native_riglogic = False
    get_addon_window_manager_properties().evaluate_dependency_graph = True
    rig_instance.start_listening()
    scene = bpy.context.scene
    frames = (1, 17, 57, 93, 120, 17)
    references = []
    for frame in frames:
        scene.frame_set(frame)
        references.append(snapshot(instance, bpy.context.evaluated_depsgraph_get()))
    switch = instance.face_board.pose.bones["CTRL_lookAtSwitch"]
    switch.location.y = 1.0
    for _iteration in range(16):
        instance.evaluate()
        bpy.context.view_layer.update()
    eye_reference = snapshot(instance, bpy.context.evaluated_depsgraph_get())
    switch.location.y = 0.0
    instance.evaluate()
    preferences.experimental_native_riglogic = True
    assert bpy.ops.character_dna.sync_native_runtime() == {"FINISHED"}
    assert engine.active(instance), controller.status()
    errors = []
    for frame, reference in zip(frames, references, strict=True):
        scene.frame_set(frame)
        actual = snapshot(instance, bpy.context.evaluated_depsgraph_get())
        errors.append(
            max(float(np.max(np.abs(values - reference[name]), initial=0)) for name, values in actual.items())
        )
    assert max(errors) < 1e-5, errors
    switch.location.y = 1.0
    instance.face_board.update_tag()
    bpy.context.view_layer.update()
    actual = snapshot(instance, bpy.context.evaluated_depsgraph_get())
    eye_error = max(float(np.max(np.abs(values - eye_reference[name]), initial=0)) for name, values in actual.items())
    assert eye_error < 1e-5, (eye_error, engine.status(instance))
    evaluated_face = instance.face_board.evaluated_get(bpy.context.evaluated_depsgraph_get())
    assert not evaluated_face.pose.bones["CTRL_C_eyesAim"].hide
    switch.location.y = 0.0
    instance.face_board.update_tag()
    bpy.context.view_layer.update()
    with controller.legacy_operation(instance):
        assert not engine.active(instance)
        instance.evaluate()
    assert engine.active(instance)
    bpy.context.scene.frame_set(17)
    actual = snapshot(instance, bpy.context.evaluated_depsgraph_get())
    resume_error = max(
        float(np.max(np.abs(values - references[-1][name]), initial=0)) for name, values in actual.items()
    )
    assert resume_error < 1e-5, resume_error
    instance.evaluate_bones = False
    scene.update_tag()
    scene.frame_set(57)
    instance.evaluate_bones = True
    scene.update_tag()
    scene.frame_set(17)
    actual = snapshot(instance, bpy.context.evaluated_depsgraph_get())
    flags_error = max(
        float(np.max(np.abs(values - references[-1][name]), initial=0)) for name, values in actual.items()
    )
    assert flags_error < 1e-5, flags_error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    blend = args.output.with_suffix(".blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), relative_remap=False)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    instance = get_active_rig_instance()
    assert engine.active(instance), controller.status()
    bpy.context.scene.frame_set(17)
    actual = snapshot(instance, bpy.context.evaluated_depsgraph_get())
    reload_error = max(
        float(np.max(np.abs(values - references[-1][name]), initial=0)) for name, values in actual.items()
    )
    assert reload_error < 1e-5, reload_error
    editor_restored = None
    if args.editor:
        from character_dna.editors.raw_control_editor.editor import RawControlEditor

        editor = RawControlEditor.for_instance(instance)
        rows = instance.raw_control_editor.raw_controls
        instance.raw_control_editor.raw_controls_active_index = next(
            index for index, row in enumerate(rows) if "jawOpen" in row.name
        )
        editor.enter(bpy.context)
        assert editor.is_editing and not engine.active(instance)
        editor.revert(bpy.context)
        assert not editor.is_editing and engine.active(instance)
        bpy.context.scene.frame_set(17)
        editor_restored = engine.status(instance)
        assert editor_restored == "Native"
    duplicate_error = None
    if args.duplicate:
        from character_dna.utilities import get_addon_scene_properties

        original_name = instance.name
        existing_names = set(get_addon_scene_properties().rig_instance_list.keys())
        before = snapshot(instance, bpy.context.evaluated_depsgraph_get())["head_pose"]
        assert bpy.ops.character_dna.duplicate_rig_instance(
            new_name="native_copy", new_folder=str(args.output.parent), copy_face_board=True
        ) == {"FINISHED"}
        instances = get_addon_scene_properties().rig_instance_list
        original = instances[original_name]
        duplicate_name = (set(instances.keys()) - existing_names).pop()
        duplicate = instances[duplicate_name]
        assert engine.active(original) and engine.active(duplicate), controller.status()
        assert engine.instance_id(original) != engine.instance_id(duplicate)
        duplicate.face_board.animation_data.action = None
        duplicate.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.9
        duplicate.face_board.update_tag()
        bpy.context.view_layer.update()
        after = snapshot(original, bpy.context.evaluated_depsgraph_get())["head_pose"]
        duplicate_error = float(np.max(np.abs(after - before)))
        assert duplicate_error < 1e-5, duplicate_error
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        instance = get_active_rig_instance()
    undo_error = redo_error = None
    if args.undo:
        bpy.context.preferences.edit.use_global_undo = True
        bpy.ops.ed.undo_push(message="Native runtime baseline")
        instance.face_board.animation_data.action = None
        instance.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.8
        instance.face_board.update_tag()
        bpy.context.view_layer.update()
        changed = snapshot(instance, bpy.context.evaluated_depsgraph_get())
        assert (
            max(float(np.max(np.abs(values - references[-1][name]), initial=0)) for name, values in changed.items())
            > 0.001
        )
        bpy.ops.ed.undo_push(message="Native runtime changed")
        bpy.ops.ed.undo()
        instance = get_active_rig_instance()
        bpy.context.view_layer.update()
        restored = snapshot(instance, bpy.context.evaluated_depsgraph_get())
        undo_error = max(
            float(np.max(np.abs(values - references[-1][name]), initial=0)) for name, values in restored.items()
        )
        assert undo_error < 1e-5, undo_error
        bpy.ops.ed.redo()
        instance = get_active_rig_instance()
        bpy.context.view_layer.update()
        restored = snapshot(instance, bpy.context.evaluated_depsgraph_get())
        redo_error = max(float(np.max(np.abs(values - changed[name]), initial=0)) for name, values in restored.items())
        assert redo_error < 1e-5, redo_error
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        instance = get_active_rig_instance()
    preferences.experimental_native_riglogic = False
    assert bpy.ops.character_dna.sync_native_runtime() == {"FINISHED"}
    assert not engine.carriers() and not engine.active(instance)
    bpy.context.scene.frame_set(57)
    actual = snapshot(instance, bpy.context.evaluated_depsgraph_get())
    disable_error = max(
        float(np.max(np.abs(values - references[2][name]), initial=0)) for name, values in actual.items()
    )
    assert disable_error < 1e-5, disable_error
    result = {
        "passed": True,
        "frame_errors": errors,
        "eye_error": eye_error,
        "resume_error": resume_error,
        "flags_error": flags_error,
        "undo_error": undo_error,
        "redo_error": redo_error,
        "duplicate_error": duplicate_error,
        "editor_restored": editor_restored,
        "reload_error": reload_error,
        "disable_error": disable_error,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
