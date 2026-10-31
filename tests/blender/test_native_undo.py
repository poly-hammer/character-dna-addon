"""Exercise Blender's real undo stack in a disposable process."""

import subprocess
import sys

from pathlib import Path

import bpy
import pytest

from character_dna.constants import ToolInfo
from character_dna.runtime import engine
from character_dna.utilities import get_active_rig_instance, remove_instance_prefix


def _instances():
    return getattr(bpy.context.scene, ToolInfo.NAME).rig_instance_list


def _snapshot():
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    values = {}
    for instance in _instances():
        for component in ("head", "body"):
            rig = getattr(instance, f"{component}_rig")
            if rig:
                for bone in rig.evaluated_get(graph).pose.bones:
                    values[f"{instance.name}/{component}/{bone.name}"] = tuple(v for row in bone.matrix for v in row)
        if instance.head_mesh and instance.head_mesh.data.shape_keys:
            keys = instance.head_mesh.data.shape_keys.evaluated_get(graph)
            values[f"{instance.name}/shapes"] = tuple(block.value for block in keys.key_blocks)
            mesh = instance.head_mesh.evaluated_get(graph).data
            values[f"{instance.name}/vertices"] = tuple(v for vertex in mesh.vertices for v in vertex.co)
        if instance.face_board:
            face = instance.face_board.evaluated_get(graph)
            values[f"{instance.name}/eyes"] = tuple(
                (bone if hasattr(bone, "hide") else bone.bone).hide
                for bone in face.pose.bones
                if bone.name.startswith("CTRL_") and "eye" in bone.name
            )
    return values


def _exercise_undo():
    root = Path(__file__).resolve().parents[2]
    bpy.ops.preferences.addon_enable(module="character_dna")
    bpy.context.preferences.addons[ToolInfo.NAME].preferences.backup_manager.enabled = False
    bpy.ops.wm.read_homefile(use_empty=True)
    for number in range(2):
        name = "ada" if number == 0 else "default"
        bpy.ops.character_dna.import_dna(
            filepath=str(root / f"tests/test_files/dna/{name}/head.dna"),
            include_body=number == 0,
            import_mesh=True,
            import_bones=True,
            import_face_board=True,
            import_shape_keys=False,
            import_materials=False,
            import_vertex_groups=True,
            **{f"import_lod{index}": index == 0 for index in range(8)},
        )
        instance = get_active_rig_instance()
        if number == 0:
            engine.discard(instance)
            mesh = instance.head_mesh
            mesh.shape_key_add(name="Basis")
            for channel in ("brow_down_L", "brow_down_R"):
                name = f"{remove_instance_prefix(mesh.name, instance.name)}__{channel}"
                mesh.shape_key_add(name=name).data[0].co.x += 0.025
            instance.destroy()
            instance.initialize()
            bpy.ops.character_dna.sync_native_runtime()
    assert len(_instances()) == 2
    assert len(engine._models) == 3
    models = dict(engine._models)
    other = bpy.data.scenes.new("OtherUndoScene")
    other_instance = getattr(other, ToolInfo.NAME).rig_instance_list.add()
    other_instance.name = "OtherUndoInstance"
    other_name = other_instance.name
    other_cache_key = other_instance.cache_key("head", "runtime_plan")
    other_instance.data[other_cache_key] = {"rig": _instances()[0].head_rig}
    bpy.context.preferences.edit.use_global_undo = True
    neutral = _snapshot()
    bpy.ops.ed.undo_push(message="Neutral characters")
    for index, instance in enumerate(_instances()):
        face = instance.face_board
        face.pose.bones["CTRL_C_jaw"].location.y = 0.7 / (index + 1)
        face.pose.bones["CTRL_L_brow_down"].location.y = 0.5
        face.pose.bones["CTRL_lookAtSwitch"].location.y = 1.0
        face.update_tag()
    changed = _snapshot()
    assert changed != neutral
    assert any(changed[key] != neutral[key] for key in changed if key.endswith("/shapes"))
    bpy.ops.ed.undo_push(message="Posed characters")
    for _ in range(4):
        for action, expected in (("undo", neutral), ("redo", changed)):
            generation = engine._generation
            old_records = dict(engine._records)
            assert getattr(bpy.ops.ed, action)() == {"FINISHED"}
            actual = _snapshot()
            assert actual.keys() == expected.keys()
            for key, values in actual.items():
                assert values == pytest.approx(expected[key], abs=1e-5), (action, key)
            assert not engine.errors(), engine.errors()
            assert engine._generation > generation
            assert engine._models.keys() == models.keys()
            assert all(engine._models[key] is model for key, model in models.items())
            assert all(record is not old for record in engine._records.values() for old in old_records.values())
            assert all(
                key[0] == engine._generation for record in engine._records.values() for key in record["contexts"]
            )
            assert not engine._undo_models and not engine._restoring_undo
            current_other = getattr(bpy.data.scenes["OtherUndoScene"], ToolInfo.NAME).rig_instance_list[other_name]
            assert other_cache_key not in current_other.data
    print("NATIVE_UNDO_REDO_PASSED", flush=True)


def test_multiple_characters_shapes_and_inactive_scene_survive_undo():
    root = Path(__file__).resolve().parents[2]
    script = (
        f"import runpy, sys; sys.path.insert(0, {str(root / 'src/addons')!r}); "
        f"runpy.run_path({str(Path(__file__).resolve())!r})['_exercise_undo']()"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(root / "tests/utilities/process.py"), script],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    assert "NATIVE_UNDO_REDO_PASSED" in result.stdout
