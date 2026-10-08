"""Editor operations must leave modern characters bound and out of migration UI."""

import queue
import shutil

from types import SimpleNamespace

import bpy
import pytest

from character_dna.runtime import controller, engine, ui_refresh
from character_dna.utilities import (
    detect_legacy_data,
    detect_runtime_migration,
    get_active_head,
    get_active_rig_instance,
)
from constants import TEST_DNA_FOLDER


shape_operators = pytest.importorskip("character_dna.editors.shape_key_editor.operators")


@pytest.fixture
def character(tmp_path):
    bpy.ops.wm.read_homefile(use_empty=True)
    for name in ("head.dna", "body.dna", "ExportManifest.json"):
        shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, tmp_path / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / "head.dna"),
        include_body=True,
        import_face_board=True,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    )
    instance = get_active_rig_instance()
    assert engine.active(instance)
    yield instance
    bpy.ops.wm.read_homefile(use_empty=True)


def assert_current_scene():
    scene = bpy.context.scene
    assert detect_legacy_data(scene) is None
    assert not detect_runtime_migration(scene)
    ui_refresh.migration_needed(scene)
    ui_refresh._refresh_migration()
    assert not ui_refresh.migration_needed(scene)


def assert_faceboard_live(instance):
    matrices = []
    for value in (0.0, 0.8):
        instance.face_board.pose.bones["CTRL_C_jaw"].location.y = value
        instance.face_board.update_tag()
        bpy.context.view_layer.update()
        rig = instance.head_rig.evaluated_get(bpy.context.view_layer.depsgraph)
        matrices.append(rig.pose.bones["FACIAL_C_Jaw"].matrix.copy())
    assert matrices[0] != matrices[1]


def import_operator():
    # Run the production modal methods without registering a timer in headless bpy.
    operator = SimpleNamespace(_timer=None, _commands_queue=queue.Queue(), _batch_size=25)
    for name in ("validate", "set_commands_queue", "finish", "modal"):
        method = getattr(shape_operators.ImportShapeKeys, name)
        setattr(operator, name, lambda *args, method=method: method(operator, *args))
    operator.report = lambda *_args: None
    return operator


def start_small_import(monkeypatch, operator):
    build = shape_operators.utilities.build_shape_key_import_commands

    def small_queue(component, commands):
        # Keep real DNA geometry and callbacks, but only allocate the first two
        # head targets so the lifecycle regression is small enough for CI.
        all_commands = queue.Queue()
        build(component, all_commands)
        for _ in range(3):
            commands.put(all_commands.get_nowait())

    monkeypatch.setattr(shape_operators, "build_shape_key_import_commands", small_queue)
    operator.set_commands_queue(bpy.context, get_active_head(), operator._commands_queue)
    operator._commands_queue_size = operator._commands_queue.qsize()


@pytest.mark.parametrize("finish_event", ["TIMER", "ESC"])
def test_shape_import_restores_live_bindings(character, monkeypatch, finish_event):
    instance = character
    assert_current_scene()
    body_state = instance.body_instance
    for _ in range(2):  # Fresh import followed by replacement of the existing Key ID.
        operator = import_operator()
        start_small_import(monkeypatch, operator)
        if finish_event == "ESC":
            operator._batch_size = 2
        assert_current_scene()
        # The first timer tick mutates shape keys; the next finishes or cancels.
        operator.modal(bpy.context, SimpleNamespace(type="TIMER"))
        assert_current_scene()
        assert operator.modal(bpy.context, SimpleNamespace(type=finish_event)) == {"FINISHED"}
        assert instance.body_instance is body_state
        assert instance.head_shape_key_blocks
        for blocks in instance.head_shape_key_blocks.values():
            for block in blocks:
                assert block.id_data.animation_data.drivers.find(block.path_from_id("value")) is not None
        assert engine.binding_issues(instance) == []
        assert_current_scene()
        assert_faceboard_live(instance)


@pytest.mark.parametrize("editor_name", ["raw_control_editor", "rbf_editor", "shape_key_editor"])
def test_failed_editor_entry_restores_runtime(character, monkeypatch, editor_name):
    from character_dna.editors.shared.editor import Editor

    editor = next(cls.for_instance(character) for cls in Editor._registry if cls.editor_id == editor_name)

    def fail(_context):
        raise RuntimeError("entry failed")

    monkeypatch.setattr(editor, "_enter", fail)
    with pytest.raises(RuntimeError, match="entry failed"):
        editor.enter(bpy.context)
    assert not controller.is_suspended(character)
    assert engine.binding_issues(character) == []
    assert_current_scene()
    assert_faceboard_live(character)


@pytest.mark.parametrize("editor_name", ["raw_control_editor", "rbf_editor", "shape_key_editor"])
@pytest.mark.parametrize("finish", ["commit", "revert"])
def test_editor_lifecycle_is_not_legacy(character, monkeypatch, editor_name, finish):
    from character_dna.editors.shared.editor import Editor

    instance = character
    if editor_name == "shape_key_editor":
        operator = import_operator()
        start_small_import(monkeypatch, operator)
        operator.modal(bpy.context, SimpleNamespace(type="TIMER"))
        operator.modal(bpy.context, SimpleNamespace(type="TIMER"))
        instance.shape_key_editor.shape_key_list_active_index = 0
    elif editor_name == "raw_control_editor":
        properties = instance.raw_control_editor
        properties.raw_controls_active_index = properties.raw_controls.find("CTRL_expressions.jawOpen")
    editor = next(cls.for_instance(instance) for cls in Editor._registry if cls.editor_id == editor_name)
    assert_current_scene()
    editor.enter(bpy.context)
    assert editor.is_editing
    assert controller.is_suspended(instance)
    assert_current_scene()
    getattr(editor, finish)(bpy.context)
    assert not editor.is_editing
    assert not controller.is_suspended(instance)
    assert engine.binding_issues(instance) == []
    assert_current_scene()
    assert_faceboard_live(instance)


def test_shape_import_failure_restores_runtime(character, monkeypatch):
    operator = import_operator()
    start_small_import(monkeypatch, operator)

    def fail(**_kwargs):
        raise RuntimeError("shape creation failed")

    commands = list(operator._commands_queue.queue)
    index, mesh_index, description, kwargs_callback, _callback = commands[1]
    commands[1] = index, mesh_index, description, kwargs_callback, fail
    operator._commands_queue = queue.Queue()
    for command in commands:
        operator._commands_queue.put(command)
    assert operator.modal(bpy.context, SimpleNamespace(type="TIMER")) == {"CANCELLED"}
    assert not controller.is_suspended(character)
    assert engine.binding_issues(character) == []
    assert_current_scene()
    assert_faceboard_live(character)


@pytest.mark.parametrize("failure_stage", ["enter", "tracking"])
def test_raw_editor_failed_entry_restores_bone_locks(character, monkeypatch, failure_stage):
    from character_dna.editors.raw_control_editor import utilities
    from character_dna.editors.raw_control_editor.editor import RawControlEditor

    editor = RawControlEditor.for_instance(character)
    bone = character.head_rig.pose.bones["FACIAL_L_EyelidUpperB1"]
    bone.lock_location = (True, False, False)

    def fail(*_args, **_kwargs):
        raise RuntimeError("failed after locking bones")

    def enter(_context):
        utilities.lock_bones_outside_joint_group(character, character.head_dna_reader, 9999)
        assert tuple(bone.lock_location) == (True, True, True)
        if failure_stage == "enter":
            fail()

    monkeypatch.setattr(editor, "_enter", enter)
    if failure_stage == "tracking":
        monkeypatch.setattr(editor.tracking, "initialize", fail)
    with pytest.raises(RuntimeError, match="failed after locking bones"):
        editor.enter(bpy.context)

    assert tuple(bone.lock_location) == (True, False, False)
    assert utilities._BONE_LOCK_SNAPSHOT not in character.head_rig
    assert not editor.is_editing
    assert not controller.is_suspended(character)


@pytest.mark.parametrize("finish", ["commit", "revert"])
def test_raw_editor_exit_restores_locks_after_cache_loss(character, finish):
    from character_dna.editors.raw_control_editor import utilities
    from character_dna.editors.raw_control_editor.editor import RawControlEditor

    editor = RawControlEditor.for_instance(character)
    editor.properties.raw_controls_active_index = editor.properties.raw_controls.find("CTRL_expressions.mouthLeft")
    bone = character.head_rig.pose.bones["FACIAL_L_EyelidUpperB1"]
    bone.lock_rotation = (False, True, False)
    editor.enter(bpy.context)
    assert tuple(bone.lock_location) == (True, True, True)
    editor.cache.clear()

    getattr(editor, finish)(bpy.context)

    assert tuple(bone.lock_location) == (False, False, False)
    assert tuple(bone.lock_rotation) == (False, True, False)
    assert utilities._BONE_LOCK_SNAPSHOT not in character.head_rig


def test_editor_shutdown_restores_bone_locks(character):
    from character_dna import editors
    from character_dna.editors.raw_control_editor import utilities

    bone = character.head_rig.pose.bones["FACIAL_L_EyelidUpperB1"]
    bone.lock_scale = (True, False, False)
    utilities.lock_bones_outside_joint_group(character, character.head_dna_reader, 9999)
    utilities.session_cache(character).clear()
    try:
        editors.unregister_runtime()
        assert tuple(bone.lock_location) == (False, False, False)
        assert tuple(bone.lock_scale) == (True, False, False)
        assert utilities._BONE_LOCK_SNAPSHOT not in character.head_rig
    finally:
        editors.register_runtime()
