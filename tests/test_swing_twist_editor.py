"""Real DNA round trips and Blender session ownership for swing/twist authoring."""

import math
import shutil

from dataclasses import replace
from pathlib import Path

import bpy
import numpy as np
import pytest

from mathutils import Quaternion

from character_dna.bindings import dna
from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle
from character_dna.runtime import controller, engine
from character_dna.utilities import get_active_rig_instance
from constants import TEST_DNA_FOLDER


core = pytest.importorskip("character_dna.editors.swing_twist_editor.core")
from character_dna.editors.shared.editor import EditorError  # noqa: E402
from character_dna.editors.swing_twist_editor import callbacks  # noqa: E402
from character_dna.editors.swing_twist_editor.editor import SwingTwistEditor  # noqa: E402


@pytest.fixture
def reader(request):
    name = getattr(request, "param", "default")
    reader = get_dna_reader(TEST_DNA_FOLDER / name / "body.dna")
    yield reader
    release_dna_handle(reader)


def evaluate(evaluator, inputs, values, lod=0):
    state, reader = evaluator.state, evaluator.reader
    for index in range(reader.getRawControlCount()):
        state.setRawControl(index, float(reader.getRawControlName(index).endswith(".qw")))
    for index, value in zip(inputs, values, strict=True):
        state.setRawControl(index, value)
    state.setLOD(lod)
    evaluator.manager.calculate(state)
    return np.array(state.getJointOutputs())


@pytest.mark.parametrize("reader", ["ada", "default", "ada_unreal"], indirect=True)
def test_setup_discovery_and_in_memory_preview(reader):
    setups = core.read_setups(reader)
    assert len(setups) == 16
    assert sum(len(s.outputs) for s in setups) == 24
    assert core.decode(core.encode(setups)) == setups
    assert setups[8].weights == (0.5, 0.0)
    draft = list(setups)
    draft[0] = replace(setups[0], weights=(0.25,))
    original = core.DraftEvaluator(reader, setups, setups, body=True)
    changed = core.DraftEvaluator(reader, setups, draft, body=True)
    try:
        quaternion = [0.2, 0.3, 0.4, math.sqrt(0.71)]
        before = evaluate(original, setups[0].inputs, quaternion)
        after = evaluate(changed, setups[0].inputs, quaternion)
        assert not np.allclose(before, after)
        assert core.read_setups(reader) == setups
        assert core.read_setups(changed.reader) == draft
        assert changed.reader.getVertexPositionCount(0) == 0
    finally:
        changed.close()
        original.close()


@pytest.mark.parametrize("mutation", ["add", "remove", "driver", "output", "nan", "range", "axis"])
def test_reject_unsupported_edits(reader, mutation):
    baseline = core.read_setups(reader)
    draft = list(baseline)
    if mutation == "add":
        draft.append(baseline[0])
    elif mutation == "remove":
        draft.pop()
    elif mutation == "driver":
        draft[0] = replace(draft[0], inputs=baseline[1].inputs)
    elif mutation == "output":
        draft[0] = replace(draft[0], outputs=baseline[1].outputs)
    elif mutation in {"nan", "range"}:
        draft[0] = replace(draft[0], weights=(float("nan") if mutation == "nan" else 1.1,))
    else:
        draft[6] = replace(draft[6], axis=(draft[6].axis + 1) % 3)
    with pytest.raises(ValueError):
        core.validate(baseline, draft)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_coupled_axes_and_canonical_roundtrip(reader, tmp_path, axis):
    baseline = core.read_setups(reader)
    draft = [replace(s, axis=axis) if s.inputs == baseline[6].inputs else s for s in baseline]
    draft[8] = replace(draft[8], weights=(0.125, 0.0))
    writer = get_dna_writer(tmp_path / "edited.dna")
    try:
        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        core.patch(writer, baseline, draft)
        writer.write()
    finally:
        release_dna_handle(writer)
    saved = get_dna_reader(tmp_path / "edited.dna")
    left = right = None
    try:
        assert core.read_setups(saved) == draft
        assert saved.getJointCount() == reader.getJointCount()
        assert saved.getRBFSolverCount() == reader.getRBFSolverCount()
        for mesh in range(reader.getMeshCount()):
            assert list(saved.getVertexPositionXs(mesh)) == list(reader.getVertexPositionXs(mesh))
        left = core.DraftEvaluator(reader, baseline, draft, body=True)
        right = core.DraftEvaluator(saved, draft, draft, body=True)
        for lod in range(reader.getLODCount()):
            for q in ([0, 0, 0, 1], [0.2, 0.3, 0.4, math.sqrt(0.71)], [0, 1, 0, 1e-7]):
                np.testing.assert_allclose(
                    evaluate(left, baseline[6].inputs, q, lod),
                    evaluate(right, baseline[6].inputs, q, lod),
                    atol=2e-5,
                )
    finally:
        if left:
            left.close()
        if right:
            right.close()
        release_dna_handle(saved)


@pytest.fixture
def character(tmp_path, request):
    bpy.ops.wm.read_homefile(use_empty=True)
    combined = getattr(request, "param", None) == "combined"
    names = ("head.dna", "body.dna", "ExportManifest.json") if combined else ("body.dna", "ExportManifest.json")
    for name in names:
        shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, tmp_path / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / ("head.dna" if combined else "body.dna")),
        include_body=combined,
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    )
    instance = get_active_rig_instance()
    if combined:
        instance.converter_merge_mode = "DNA"
        instance.output.folder_path = str(tmp_path / "combined")
        assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    instance.editor_component = "body"
    yield instance
    bpy.ops.wm.read_homefile(use_empty=True)


@pytest.mark.parametrize("finish", ["commit", "revert", "noop"])
def test_editor_preview_and_lifecycle(character, finish):
    instance = character
    editor = SwingTwistEditor.for_instance(instance)
    path = Path(instance.edit_target.dna_file_path)
    before = path.read_bytes()
    original = core.read_setups(instance.edit_target.dna_reader)
    pose = np.array(instance.body_rig.pose.bones["calf_l"].matrix_basis)
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
    editor.enter(bpy.context)
    assert controller.is_suspended(instance)
    assert not bpy.context.scene.tool_settings.use_keyframe_insert_auto
    assert len(instance.swing_twist_editor.setups) == 16
    instance.editor_component = "head"
    assert instance.editor_component == "body"
    instance.swing_twist_editor.preview_offset = math.radians(45)
    output = instance.body_rig.pose.bones[original[0].output_names[0]]
    initial_output = np.array(output.matrix_basis)
    if finish != "noop":
        instance.swing_twist_editor.setups[0].outputs[0].weight = 0.25
        assert not instance.swing_twist_editor.error
        assert not np.allclose(output.matrix_basis, initial_output)
    assert path.read_bytes() == before
    getattr(editor, "revert" if finish == "revert" else "commit")(bpy.context)
    assert not editor.is_editing
    assert not controller.is_suspended(instance)
    assert engine.binding_issues(instance) == []
    assert bpy.context.scene.tool_settings.use_keyframe_insert_auto
    np.testing.assert_allclose(instance.body_rig.pose.bones["calf_l"].matrix_basis, pose, atol=2e-5)
    if finish == "commit":
        saved = core.read_setups(instance.edit_target.dna_reader)
        assert saved[0].weights == (0.25,)
        assert saved[1:] == original[1:]
        assert list(instance.body_rig.pose.bones[original[0].driver]["swing_blend_weights"]) == [0.25]
    else:
        assert path.read_bytes() == before


def test_shared_axis_blend_reset_and_cache_recovery(character):
    editor = SwingTwistEditor.for_instance(character)
    character.swing_twist_editor.active_index = 6
    editor.enter(bpy.context)
    properties = character.swing_twist_editor
    properties.setups[6].axis = "Y"
    assert properties.setups[14].axis == "Y"
    properties.setups[6].outputs[0].weight = 0.25
    character.destroy()
    properties.setups[6].outputs[0].weight = 0.35
    assert not properties.error
    assert properties.setups[6].outputs[0].weight == pytest.approx(0.35)
    assert editor.validate_commit(bpy.context)[0]
    assert bpy.ops.character_dna.reset_swing_twist_blend(output_index=0) == {"FINISHED"}
    baseline = core.decode(properties.baseline_json)
    assert properties.setups[6].outputs[0].weight == pytest.approx(baseline[6].weights[0])
    assert properties.setups[6].axis == properties.setups[14].axis == "Y"
    properties.setups[6].axis = "XYZ"[baseline[6].axis]
    assert callbacks.draft(character) == baseline
    editor.revert(bpy.context)


def test_failed_commit_retains_draft_and_rolls_back(character, monkeypatch):
    from character_dna.dna_io.coordinates import CanonicalDNAWriter

    editor = SwingTwistEditor.for_instance(character)
    path = Path(character.edit_target.dna_file_path)
    original = path.read_bytes()
    editor.enter(bpy.context)
    character.swing_twist_editor.setups[0].outputs[0].weight = 0.25
    write = CanonicalDNAWriter.write

    def fail(writer):
        write(writer)
        raise RuntimeError("publication failed")

    with monkeypatch.context() as patch:
        patch.setattr(CanonicalDNAWriter, "write", fail)
        with pytest.raises(EditorError, match="publication failed"):
            editor.commit(bpy.context)
    assert editor.is_editing
    assert path.read_bytes() == original
    assert character.swing_twist_editor.setups[0].outputs[0].weight == 0.25
    editor.commit(bpy.context)
    assert not editor.is_editing


def test_external_edit_is_rejected(character):
    editor = SwingTwistEditor.for_instance(character)
    editor.enter(bpy.context)
    properties = character.swing_twist_editor
    properties.source_revision = "stale revision"
    with pytest.raises(EditorError, match="changed on disk"):
        editor.commit(bpy.context)
    assert editor.is_editing
    editor.revert(bpy.context)


@pytest.mark.parametrize("character", ["combined"], indirect=True)
@pytest.mark.parametrize("fail_first", [False, True])
def test_combined_commit_preserves_head_and_surface(character, monkeypatch, fail_first):
    from character_dna.dna_io import document
    from character_dna.editors.converter import scene_merge

    path = Path(character["combined_dna_path"])
    before = path.read_bytes()
    original_head = document.read_document(path)[1]["head"]
    proxy = scene_merge.proxy_for(character)
    identity = proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer()
    vertices = np.array([tuple(vertex.co) for vertex in proxy.data.vertices])
    editor = SwingTwistEditor.for_instance(character)
    character.swing_twist_editor.active_index = 8
    editor.enter(bpy.context)
    character.swing_twist_editor.setups[8].outputs[1].weight = 0.2
    if fail_first:

        def fail(*_args, **_kwargs):
            raise RuntimeError("combined publication failed")

        with monkeypatch.context() as patch:
            patch.setattr(document, "_publish_edit", fail)
            with pytest.raises(RuntimeError, match="combined publication failed"):
                editor.commit(bpy.context)
        assert editor.is_editing
        assert path.read_bytes() == before
        assert character.swing_twist_editor.setups[8].outputs[1].weight == pytest.approx(0.2)
    editor.commit(bpy.context)
    assert document.read_document(path)[1]["head"] == original_head
    assert core.read_setups(character.edit_target.dna_reader)[8].weights == pytest.approx((0.5, 0.2))
    assert identity == (proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer())
    np.testing.assert_array_equal(vertices, [tuple(vertex.co) for vertex in proxy.data.vertices])
    assert engine.binding_issues(character) == []


@pytest.mark.parametrize("finish", ["commit", "revert"])
def test_saved_session_recovers_draft_and_context(character, tmp_path, finish):
    path = Path(character.edit_target.dna_file_path)
    before = path.read_bytes()
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
    SwingTwistEditor.for_instance(character).enter(bpy.context)
    character.swing_twist_editor.setups[0].outputs[0].weight = 0.25
    character.swing_twist_editor.preview_offset = math.radians(-60)
    preview_pose = np.array(character.body_rig.pose.bones["calf_l"].matrix_basis)
    blend = tmp_path / "editing.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    restored = get_active_rig_instance()
    assert restored.swing_twist_editor.is_editing
    assert restored.swing_twist_editor.setups[0].outputs[0].weight == 0.25
    assert restored.swing_twist_editor.preview_offset == pytest.approx(math.radians(-60))
    restored.swing_twist_editor.setups[0].outputs[0].weight = 0.3
    assert not restored.swing_twist_editor.error
    np.testing.assert_allclose(restored.body_rig.pose.bones["calf_l"].matrix_basis, preview_pose, atol=2e-5)
    editor = SwingTwistEditor.for_instance(restored)
    getattr(editor, finish)(bpy.context)
    assert not controller.is_suspended(restored)
    assert bpy.context.scene.tool_settings.use_keyframe_insert_auto
    assert engine.binding_issues(restored) == []
    if finish == "revert":
        assert path.read_bytes() == before
    else:
        assert core.read_setups(restored.edit_target.dna_reader)[0].weights == pytest.approx((0.3,))


@pytest.mark.parametrize("rbfs", [True, False])
def test_preview_matches_committed_native_runtime(character, rbfs):
    editor = SwingTwistEditor.for_instance(character)
    character.evaluate_rbfs = rbfs
    properties = character.swing_twist_editor
    properties.active_index = 6
    rig = character.body_rig
    for name in ("calf_l", "upperarm_l"):
        bone = rig.pose.bones[name]
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = Quaternion((0.9, 0.1, 0.2, 0.3)).normalized()
    rig.update_tag()
    bpy.context.view_layer.update()
    editor.enter(bpy.context)
    properties.setups[6].outputs[0].weight = 0.25
    properties.setups[6].axis = "Z"
    assert not properties.error
    inputs = {name: rig.pose.bones[name].matrix_basis.copy() for name in ("calf_l", "upperarm_l")}
    names = [bone.name for bone in rig.pose.bones]
    expected = {name: np.array(rig.pose.bones[name].matrix) for name in names}
    editor.commit(bpy.context)
    for name, basis in inputs.items():
        rig.pose.bones[name].matrix_basis = basis
    rig.update_tag()
    bpy.context.view_layer.update()
    evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
    for name in names:
        np.testing.assert_allclose(evaluated.pose.bones[name].matrix, expected[name], atol=2e-5, err_msg=name)


def test_calf_back_90_preview_matches_complete_rbf_pose(character):
    properties = character.swing_twist_editor
    rig = character.body_rig
    rbf = character.rbf_editor
    solver_index = rbf.rbf_solver_list.find("calf_l_UERBFSolver")
    assert solver_index >= 0
    rbf.rbf_solver_list_active_index = solver_index
    solver = rbf.rbf_solver_list[solver_index]
    solver.poses_active_index = solver.poses.find("calf_l_back_90")
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    posed = rig.evaluated_get(graph)
    expected = {bone.name: np.array(bone.matrix) for bone in posed.pose.bones}
    driver_pose = np.array(posed.pose.bones["calf_l"].matrix_basis)
    mesh = character.body_mesh.evaluated_get(graph)
    vertices = np.empty(len(mesh.data.vertices) * 3, dtype=np.float32)
    mesh.data.vertices.foreach_get("co", vertices)

    solver.poses_active_index = solver.poses.find("default")
    properties.active_index = next(
        index for index, setup in enumerate(properties.setups) if setup.driver == "calf_l" and setup.kind == "Swing"
    )
    editor = SwingTwistEditor.for_instance(character)
    editor.enter(bpy.context)
    properties.preview_offset = math.radians(90)
    assert not properties.error
    graph = bpy.context.evaluated_depsgraph_get()
    preview = rig.evaluated_get(graph)
    np.testing.assert_allclose(preview.pose.bones["calf_l"].matrix_basis, driver_pose, atol=2e-5)
    # Include the RBF knee correctives, not just the selected swing/twist outputs.
    for name in ("calf_kneeBack_l", "calf_knee_l", *expected):
        np.testing.assert_allclose(preview.pose.bones[name].matrix, expected[name], atol=2e-5, err_msg=name)
    mesh = character.body_mesh.evaluated_get(graph)
    actual = np.empty_like(vertices)
    mesh.data.vertices.foreach_get("co", actual)
    np.testing.assert_allclose(actual, vertices, atol=2e-5)
    editor.revert(bpy.context)


@pytest.mark.parametrize("finish", ["commit", "revert"])
def test_preview_updates_rbf_shape_keys_and_restores_values(character, finish):
    from character_dna.editors.rbf_editor import shapes
    from character_dna.editors.rbf_editor.editor import RBFEditor

    rbf = character.rbf_editor
    solver_index = rbf.rbf_solver_list.find("calf_l_UERBFSolver")
    rbf.rbf_solver_list_active_index = solver_index
    editor = RBFEditor.for_instance(character)
    editor.enter(bpy.context)
    solver = rbf.rbf_solver_list[solver_index]
    solver.poses_active_index = solver.poses.find("calf_l_back_90")
    mesh_name = character.edit_target.dna_reader.getMeshName(0)
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="NEW", mesh_name=mesh_name, new_name="knee_preview_test"
    ) == {"FINISHED"}
    editor.commit(bpy.context)
    solver = rbf.rbf_solver_list[solver_index]
    solver.poses_active_index = solver.poses.find("default")
    mesh = character.edit_target.mesh_index_lookup[0]
    key = shapes.blocks(mesh)[f"{mesh_name}__knee_preview_test"]
    character.auto_evaluate_body = False
    key.value = 0.37

    properties = character.swing_twist_editor
    properties.active_index = 0
    editor = SwingTwistEditor.for_instance(character)
    editor.enter(bpy.context)
    assert key.value == pytest.approx(0, abs=1e-5)
    properties.preview_offset = math.radians(90)
    assert not properties.error
    assert key.value == pytest.approx(1, abs=1e-5)
    properties.setups[0].outputs[0].weight = 0.25
    assert key.value == pytest.approx(1, abs=1e-5)
    assert bpy.ops.character_dna.reset_swing_twist_preview() == {"FINISHED"}
    assert key.value == pytest.approx(0, abs=1e-5)
    properties.preview_offset = math.radians(90)
    getattr(editor, finish)(bpy.context)
    assert not character.auto_evaluate_body
    assert key.value == pytest.approx(0.37, abs=1e-5)


def test_read_only_records_survive_ui_roundtrip(character):
    baseline = core.read_setups(character.edit_target.dna_reader)
    baseline[0] = replace(baseline[0], axis=9, weights=(0.25, 0.75))
    callbacks.populate(character, baseline)
    character.swing_twist_editor.baseline_json = core.encode(baseline)
    character.swing_twist_editor.session_index = 1
    assert character.swing_twist_editor.setups[0].warning
    character.swing_twist_editor.setups[1].outputs[0].weight = 0.25
    draft = callbacks.draft(character)
    assert draft[0] == baseline[0]
    core.validate(baseline, draft)
    assert len(core.changes(baseline, draft)) == 1


def test_parameter_undo_redo_restores_scene_values(character):
    from character_dna.utilities import switch_to_object_mode

    driver = character.body_rig.pose.bones["calf_l"]
    driver.rotation_quaternion = Quaternion((0.9, 0.1, 0.2, 0.3)).normalized()
    character.body_rig.update_tag()
    bpy.context.view_layer.update()
    SwingTwistEditor.for_instance(character).enter(bpy.context)
    switch_to_object_mode()
    bpy.context.preferences.edit.use_global_undo = True
    original = character.swing_twist_editor.setups[0].outputs[0].weight
    output_name = character.swing_twist_editor.setups[0].outputs[0].name
    original_pose = np.array(character.body_rig.pose.bones[output_name].matrix_basis)
    bpy.ops.ed.undo_push(message="Original swing blend")
    character.swing_twist_editor.setups[0].outputs[0].weight = 0.25
    changed_pose = np.array(character.body_rig.pose.bones[output_name].matrix_basis)
    assert not np.allclose(original_pose, changed_pose)
    bpy.ops.ed.undo_push(message="Edited swing blend")
    assert bpy.ops.ed.undo.poll(), "This test requires Blender's undo stack"
    for operation, expected in (("undo", original), ("redo", 0.25)):
        assert getattr(bpy.ops.ed, operation)() == {"FINISHED"}
        restored = get_active_rig_instance()
        assert restored.swing_twist_editor.is_editing
        assert restored.swing_twist_editor.setups[0].outputs[0].weight == expected
        assert controller.is_suspended(restored)
        bpy.context.view_layer.update()
        rig = restored.body_rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
        np.testing.assert_allclose(
            rig.pose.bones[output_name].matrix_basis,
            original_pose if operation == "undo" else changed_pose,
            atol=2e-5,
        )
    SwingTwistEditor.for_instance(get_active_rig_instance()).revert(bpy.context)


def test_multiple_character_drafts_and_auto_key_are_independent(character, tmp_path):
    from character_dna.constants import ToolInfo

    instances = getattr(bpy.context.scene, ToolInfo.NAME).rig_instance_list
    first_name = character.name
    other_path = tmp_path / "other"
    other_path.mkdir()
    for name in ("body.dna", "ExportManifest.json"):
        shutil.copyfile(TEST_DNA_FOLDER / "default" / name, other_path / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(other_path / "body.dna"),
        include_body=False,
        import_face_board=False,
        import_mesh=False,
        import_bones=True,
        import_materials=False,
    )
    second = get_active_rig_instance()
    second.editor_component = "body"
    first = instances[first_name]
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
    left, right = SwingTwistEditor.for_instance(first), SwingTwistEditor.for_instance(second)
    left.enter(bpy.context)
    right.enter(bpy.context)
    first.swing_twist_editor.setups[0].outputs[0].weight = 0.25
    second.swing_twist_editor.setups[0].outputs[0].weight = 0.75
    assert not first.swing_twist_editor.error
    assert not second.swing_twist_editor.error
    assert first.swing_twist_editor.setups[0].outputs[0].weight == 0.25
    assert second.swing_twist_editor.setups[0].outputs[0].weight == 0.75
    left.revert(bpy.context)
    assert right.is_editing
    assert not bpy.context.scene.tool_settings.use_keyframe_insert_auto
    right.revert(bpy.context)
    assert bpy.context.scene.tool_settings.use_keyframe_insert_auto
    assert engine.binding_issues(first) == engine.binding_issues(second) == []


def test_editor_adds_no_app_handlers_and_shutdown_restores_scene(character):
    from character_dna import editors

    def handlers():
        return {
            name: tuple(value)
            for name in dir(bpy.app.handlers)
            if isinstance(value := getattr(bpy.app.handlers, name), list)
        }

    before = handlers()
    try:
        SwingTwistEditor.for_instance(character).enter(bpy.context)
        character.swing_twist_editor.setups[0].outputs[0].weight = 0.25
        assert handlers() == before
        assert not any("swing_twist_editor" in callback.__module__ for values in before.values() for callback in values)
        editors.unregister_runtime()
        assert not character.swing_twist_editor.is_editing
        assert not controller.is_suspended(character)
        assert engine.binding_issues(character) == []
    finally:
        editors.register_runtime()


def test_selection_and_writeback_are_scoped_to_one_setup(character):
    properties = character.swing_twist_editor
    properties.active_index = 8
    editor = SwingTwistEditor.for_instance(character)
    original = core.read_setups(character.edit_target.dna_reader)
    editor.enter(bpy.context)
    properties.active_index = 0
    assert properties.active_index == properties.session_index == 8
    properties.setups[0].outputs[0].weight = 0.1
    assert properties.setups[0].outputs[0].weight == original[0].weights[0]
    properties.setups[8].outputs[1].weight = 0.2
    assert not properties.error
    editor.commit(bpy.context)
    saved = core.read_setups(character.edit_target.dna_reader)
    assert saved[8].weights == pytest.approx((0.5, 0.2))
    assert saved[:8] == original[:8] and saved[9:] == original[9:]
    properties.active_index = 0
    assert properties.active_index == 0


@pytest.mark.parametrize("index", [0, 8, 6, 14])
def test_blend_and_axis_callbacks_update_bones_immediately(character, index):
    properties = character.swing_twist_editor
    properties.active_index = index
    original = core.read_setups(character.edit_target.dna_reader)
    selected = original[index]
    rig = character.body_rig
    editor = SwingTwistEditor.for_instance(character)
    editor.enter(bpy.context)
    properties.preview_offset = math.radians(45)
    output = rig.pose.bones[selected.output_names[0]]
    unrelated = rig.pose.bones[next(s for s in original if s.driver != selected.driver).output_names[0]]
    unrelated_pose = np.array(unrelated.matrix_basis)
    # Every update refreshes the complete evaluated pose, including other setups.
    unrelated.location.x += 0.1
    row = properties.setups[index]
    row.outputs[0].weight = 0.1
    low = np.array(output.matrix_basis)
    row.outputs[0].weight = 0.8
    assert not properties.error
    assert not np.allclose(output.matrix_basis, low)
    rotations = []
    for axis in "XYZ":
        row.axis = axis
        assert not properties.error
        rotations.append(np.array(output.matrix_basis))
    assert all(not np.allclose(rotations[a], rotations[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    np.testing.assert_allclose(unrelated.matrix_basis, unrelated_pose, atol=1e-6)
    assert core.read_setups(character.edit_target.dna_reader) == original
    editor.revert(bpy.context)


@pytest.mark.parametrize("index,collection_name", [(0, "Swings"), (8, "Twists")])
@pytest.mark.parametrize("finish", ["commit", "revert"])
def test_collection_visibility_is_revealed_then_restored(character, index, collection_name, finish):
    properties = character.swing_twist_editor
    properties.active_index = index
    rig = character.body_rig
    collection = rig.data.collections_all[collection_name]
    collection.is_visible = False
    rig.data.collections_all["Drivers"].is_solo = True
    bone = rig.data.bones[properties.setups[index].outputs[0].name]
    bone.hide = True
    parent = rig.data.collections.new("Hidden parent")
    nested = rig.data.collections.new("Nested output", parent=parent)
    nested.assign(bone)
    parent.is_visible = nested.is_visible = False
    before = {c.name: (c.is_visible, c.is_solo) for c in rig.data.collections_all}
    editor = SwingTwistEditor.for_instance(character)
    editor.enter(bpy.context)
    assert {c.name for c in rig.data.collections_all if c.is_visible} == {"Drivers", collection_name}
    assert not parent.is_visible and not nested.is_visible
    assert not any(c.is_solo for c in rig.data.collections_all)
    assert not bone.hide
    assert bpy.ops.character_dna.select_swing_twist_bone(bone_name=bone.name) == {"FINISHED"}
    assert {c.name for c in rig.data.collections_all if c.is_visible} == {"Drivers", collection_name}
    properties.setups[index].outputs[0].weight = 0.25
    getattr(editor, finish)(bpy.context)
    assert before == {c.name: (c.is_visible, c.is_solo) for c in rig.data.collections_all}
    assert bone.hide


def test_failed_entry_restores_collection_visibility(character, monkeypatch):
    from character_dna.editors.swing_twist_editor import editor as editor_module

    rig = character.body_rig
    before = {c.name: (c.is_visible, c.is_solo) for c in rig.data.collections_all}

    def fail(_instance):
        assert rig.data.collections_all["Swings"].is_visible
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(editor_module, "apply_selected", fail)
    editor = SwingTwistEditor.for_instance(character)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        editor.enter(bpy.context)
    assert not editor.is_editing
    assert before == {c.name: (c.is_visible, c.is_solo) for c in rig.data.collections_all}
    assert engine.binding_issues(character) == []


@pytest.mark.parametrize("index", [0, 8])
@pytest.mark.parametrize("finish", ["commit", "revert"])
def test_offset_is_absolute_preview_only_and_restores_pose(character, index, finish):
    properties = character.swing_twist_editor
    properties.active_index = index
    baseline = core.read_setups(character.edit_target.dna_reader)
    path = Path(character.edit_target.dna_file_path)
    before = path.read_bytes()
    driver = character.body_rig.pose.bones[baseline[index].driver]
    original_pose = np.array(driver.matrix_basis)
    properties.preview_offset = math.radians(10)
    editor = SwingTwistEditor.for_instance(character)
    editor.enter(bpy.context)
    assert properties.preview_offset == 0
    np.testing.assert_allclose(driver.matrix_basis, original_pose, atol=2e-5)
    properties.preview_offset = math.radians(45)
    initial_pose = np.array(driver.matrix_basis)
    output = character.body_rig.pose.bones[baseline[index].output_names[0]]
    initial_output = np.array(output.matrix_basis)
    assert not np.allclose(initial_pose, original_pose)
    properties.preview_offset = math.radians(-70)
    assert not properties.error
    assert not np.allclose(output.matrix_basis, initial_output)
    for degrees in (10, 90, -30, 45):
        properties.preview_offset = math.radians(degrees)
        assert not properties.error
    # Re-evaluation from the entry pose must never add the same offset twice.
    np.testing.assert_allclose(driver.matrix_basis, initial_pose, atol=2e-5)
    np.testing.assert_allclose(output.matrix_basis, initial_output, atol=2e-5)
    assert bpy.ops.character_dna.reset_swing_twist_preview() == {"FINISHED"}
    assert properties.preview_offset == 0
    assert not properties.error
    np.testing.assert_allclose(driver.matrix_basis, original_pose, atol=2e-5)
    properties.preview_offset = math.radians(90)
    assert callbacks.draft(character) == baseline
    assert path.read_bytes() == before
    getattr(editor, finish)(bpy.context)
    assert path.read_bytes() == before
    np.testing.assert_allclose(driver.matrix_basis, original_pose, atol=2e-5)
    assert core.read_setups(character.edit_target.dna_reader) == baseline


def test_preview_offset_makes_every_setup_blend_visible(character):
    properties = character.swing_twist_editor
    for index in range(len(properties.setups)):
        properties.active_index = index
        editor = SwingTwistEditor.for_instance(character)
        editor.enter(bpy.context)
        properties.preview_offset = math.radians(45)
        row = properties.setups[index]
        output = character.body_rig.pose.bones[row.outputs[0].name]
        row.outputs[0].weight = 0.1
        low = np.array(output.matrix_basis)
        row.outputs[0].weight = 0.8
        assert not properties.error, row.name
        assert not np.allclose(output.matrix_basis, low), row.name
        editor.revert(bpy.context)
