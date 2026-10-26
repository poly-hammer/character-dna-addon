"""Native scalar routing and the user-facing driven-item lifecycle."""

from dataclasses import asdict

import bpy
import numpy as np
import pytest

from character_dna.bindings import dna, riglogic
from character_dna.dna_io import release_dna_handle
from character_dna.dna_io.coordinates import configuration
from character_dna.editors.rbf_editor import core, shapes, utilities
from character_dna.editors.rbf_editor.editor import RBFEditor
from test_component_editing import component_reader as _component_reader, solver_data
from test_merge_split import character as _character


component_reader = _component_reader
character = _character


def serialize(reader, data, targets=()):
    stream = dna.MemoryStream()
    writer = dna.BinaryStreamWriter(stream)
    core.commit_rbf_data_to_dna(reader, writer, data, shape_targets=list(targets))
    stream.seek(0)
    result = dna.BinaryStreamReader(stream, configuration(), None)
    result.read()
    assert dna.Status.isOk(), dna.Status.get().message
    return stream, writer, result


def evaluate(reader, solver, pose_number):
    manager = riglogic.RigLogic(reader, riglogic.Configuration(), None)
    state = riglogic.RigInstance(rigLogic=manager, memRes=None)
    indices = list(reader.getRBFSolverRawControlIndices(solver))
    values = list(reader.getRBFSolverRawControlValues(solver))
    for index, value in zip(
        indices, values[pose_number * len(indices) : (pose_number + 1) * len(indices)], strict=True
    ):
        state.setRawControl(index, value)
    manager.calculate(state)
    result = (np.array(state.getBlendShapeOutputs()), np.array(state.getJointOutputs()))
    release_dna_handle(state)
    release_dna_handle(manager)
    return result


def test_native_shape_coefficients_roundtrip_and_unlink(component_reader):
    _, reader = component_reader
    data = solver_data(reader)
    solver = data[0]
    name = "test_rbf_shape"
    for number, pose in enumerate(solver["poses"]):
        item = asdict(core.initialize_rbf_driven_data(name, pose["pose_index"], -1))
        item.update(data_type="SHAPE_KEY", scalar_value=0.7 if number == 1 else 0.0, joint_index=-1)
        pose["driven"].append(item)
    handles = serialize(reader, data, [{"name": name, "mesh_index": 0, "vertices": [0], "deltas": [[0.1, 0, 0]]}])
    result = handles[-1]
    extra = ()
    try:
        channel = result.getBlendShapeChannelCount() - 1
        assert result.getBlendShapeChannelName(channel) == name
        for pose_number in (0, 1, 2):
            before, bones_before = evaluate(reader, 0, pose_number)
            after, bones_after = evaluate(result, 0, pose_number)
            np.testing.assert_allclose(bones_after, bones_before, atol=1e-6)
            np.testing.assert_allclose(after[:-1], before, atol=1e-6)
            assert after[-1] == pytest.approx(0.7 if pose_number == 1 else 0.0, abs=1e-5)
        restored = solver_data(result)
        entry = next(x for x in restored[0]["poses"][1]["driven"] if x["name"] == name)
        assert entry["scalar_value"] == pytest.approx(0.7)
        for pose in restored[0]["poses"]:
            pose["driven"] = [item for item in pose["driven"] if item["name"] != name]
        extra = serialize(result, restored)
        assert channel not in extra[-1].getBlendShapeChannelOutputIndices()
        assert extra[-1].getBlendShapeChannelName(channel) == name
        assert extra[-1].getBlendShapeTargetCount(0) == result.getBlendShapeTargetCount(0)
    finally:
        for handle in reversed(extra + handles):
            release_dna_handle(handle)


def test_repeated_weight_edits_reuse_control_and_support_shape_only(component_reader):
    _, reader = component_reader
    data = solver_data(reader)
    for number, pose in enumerate(data[0]["poses"]):
        item = asdict(core.initialize_rbf_driven_data("scalar_only", pose["pose_index"], -1))
        item.update(data_type="SHAPE_KEY", scalar_value=0.25 if number == 1 else 0.0, joint_index=-1)
        pose["driven"] = [item]
    handles = serialize(reader, data, [{"name": "scalar_only", "mesh_index": 0, "vertices": [], "deltas": []}])
    revised = ()
    try:
        saved = handles[-1]
        authored = solver_data(saved)
        for pose in authored[0]["poses"]:
            # Zero bone rows may still describe joint-group membership; they do not contribute.
            pose["driven"] = [item for item in pose["driven"] if item["data_type"] != "BONE"]
        scalar = next(item for item in authored[0]["poses"][1]["driven"] if item["name"] == "scalar_only")
        scalar["scalar_value"] = 0.6
        revised = serialize(saved, authored)
        assert revised[-1].getRBFPoseControlCount() == saved.getRBFPoseControlCount()
        assert evaluate(revised[-1], 0, 1)[0][-1] == pytest.approx(0.6, abs=1e-5)
        for item in core.get_rbf_driven_data(revised[-1], authored[0]["poses"][1]["pose_index"]):
            np.testing.assert_allclose(item.location, 0, atol=1e-6)
            np.testing.assert_allclose(item.euler_rotation, 0, atol=1e-6)
    finally:
        for handle in reversed(revised + handles):
            release_dna_handle(handle)


def test_candidates_exclude_all_dna_drivers(component_reader):
    from types import SimpleNamespace

    _, reader = component_reader
    mesh = bpy.data.meshes.new("candidate_test")
    obj = bpy.data.objects.new("candidate_test", mesh)
    obj.shape_key_add(name="Basis", from_mix=False)
    obj.shape_key_add(name="unassigned", from_mix=False)
    occupied = obj.shape_key_add(name="scene_driven", from_mix=False)
    occupied.driver_add("value").driver.expression = "0.0"
    instance = SimpleNamespace(
        edit_target=SimpleNamespace(dna_reader=reader, mesh_index_lookup={0: obj}),
        rbf_editor=SimpleNamespace(rbf_solver_list=core.get_rbf_solver_data(reader)),
    )
    try:
        available = shapes.candidates(instance, 0)
        assert "unassigned" in available
        assert "scene_driven" not in available
        for channel in reader.getBlendShapeChannelOutputIndices():
            assert reader.getBlendShapeChannelName(channel) not in available
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def enter(instance, kind="body"):
    instance.editor_component = kind
    instance.edit_target.initialize()
    editor = RBFEditor.for_instance(instance)
    editor.enter(bpy.context)
    solver = utilities.get_active_solver(instance)
    solver.poses_active_index = 1
    return editor, solver


def test_create_delete_and_revert_keys(character):
    from character_dna.shape_names import stored_name

    instance = character
    editor, solver = enter(instance)
    reader = instance.edit_target.dna_reader
    obj = instance.edit_target.mesh_index_lookup[0]
    original_keys = list(shapes.blocks(obj))
    name = "test_calf_corrective__" + "long_name_" * 8
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="NEW", mesh_name=reader.getMeshName(0), new_name=name
    ) == {"FINISHED"}
    full_name = stored_name(obj.data.shape_keys, f"{reader.getMeshName(0)}__{name}")
    assert full_name in shapes.blocks(obj)
    assert shapes.blocks(obj)[full_name].value == pytest.approx(1.0, abs=1e-5)
    assert [p.driven[-1].scalar_value for p in solver.poses] == [0, 1] + [0] * (len(solver.poses) - 2)
    assert name not in shapes.candidates(instance, 0)
    pose = utilities.get_active_pose(instance)
    pose.driven_active_index = len(pose.driven) - 1
    assert bpy.ops.character_dna.remove_rbf_driven(delete_shape_keys=True) == {"FINISHED"}
    assert full_name not in shapes.blocks(obj)
    editor.revert(bpy.context)
    assert list(shapes.blocks(obj)) == original_keys


def test_new_shape_commit_then_delete_revert(character, monkeypatch):
    instance = character
    editor, _ = enter(instance)
    mesh_name = instance.edit_target.dna_reader.getMeshName(0)
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="NEW", mesh_name=mesh_name, new_name="test_corrective"
    ) == {"FINISHED"}
    commit = core.commit_rbf_data_to_dna
    monkeypatch.setattr(
        core, "commit_rbf_data_to_dna", lambda **_kwargs: (_ for _ in ()).throw(ValueError("Simulated write failure"))
    )
    with pytest.raises(Exception, match="Simulated write failure"):
        editor.commit(bpy.context)
    assert editor.is_editing
    assert utilities.get_active_pose(instance).driven[-1].name == "test_corrective"
    monkeypatch.setattr(core, "commit_rbf_data_to_dna", commit)
    editor.commit(bpy.context)
    reader = instance.edit_target.dna_reader
    assert "test_corrective" in [reader.getBlendShapeChannelName(i) for i in range(reader.getBlendShapeChannelCount())]
    obj = instance.edit_target.mesh_index_lookup[0]
    assert shapes.SNAPSHOT not in obj
    assert any(item.name == f"{mesh_name}__test_corrective" for item in instance.shape_key_editor.shape_key_list)
    editor, _solver = enter(instance)
    pose = utilities.get_active_pose(instance)
    pose.driven_active_index = next(i for i, item in enumerate(pose.driven) if item.name == "test_corrective")
    assert bpy.ops.character_dna.remove_rbf_driven(delete_shape_keys=True) == {"FINISHED"}
    assert f"{mesh_name}__test_corrective" not in shapes.blocks(obj)
    editor.revert(bpy.context)
    assert f"{mesh_name}__test_corrective" in shapes.blocks(obj)


def test_available_keys_rollback_and_pose_copy(character, monkeypatch):
    from character_dna.editors.rbf_editor import change_tracker, preview

    instance = character
    editor, solver = enter(instance)
    obj = instance.edit_target.mesh_index_lookup[0]
    mesh_name = instance.edit_target.dna_reader.getMeshName(0)
    obj.shape_key_add(name="Basis", from_mix=False)
    original = obj.shape_key_add(name="unassigned", from_mix=False)
    original.value = 0.0
    original.data[0].co.x += 0.01
    assert "unassigned" in shapes.candidates(instance, 0)
    show = preview.show
    monkeypatch.setattr(preview, "show", lambda *_args: (_ for _ in ()).throw(ValueError("Simulated preview failure")))
    with pytest.raises(RuntimeError, match="Simulated preview failure"):
        bpy.ops.character_dna.add_rbf_driven(
            data_type="SHAPE_KEY", shape_source="EXISTING", mesh_name=mesh_name, shape_name="unassigned"
        )
    assert "unassigned" in shapes.blocks(obj)

    assert f"{mesh_name}__unassigned" not in shapes.blocks(obj)
    assert not any(item.data_type == "SHAPE_KEY" for pose in solver.poses for item in pose.driven)
    monkeypatch.setattr(preview, "show", show)
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="EXISTING", mesh_name=mesh_name, shape_name="unassigned"
    ) == {"FINISHED"}
    pose = utilities.get_active_pose(instance)
    pose.driven[-1].scalar_value = 0.4
    assert change_tracker.get_change_tracker(instance).change_count > 0
    success, message, _ = utilities.add_rbf_pose(instance, "test_copy", from_pose=pose)
    assert success, message
    assert utilities.get_active_pose(instance).driven[-1].scalar_value == pytest.approx(0.4)
    editor.revert(bpy.context)
    assert "unassigned" in shapes.blocks(obj)


def test_add_shape_blender_undo(character):
    from character_dna.shape_names import stored_name
    from character_dna.utilities import get_active_rig_instance, switch_to_object_mode

    instance = character
    enter(instance)
    mesh_name = instance.edit_target.dna_reader.getMeshName(0)
    switch_to_object_mode()
    bpy.context.preferences.edit.use_global_undo = True
    bpy.ops.ed.undo_push(message="Before RBF shape")
    name = "undo_corrective_" * 7
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="NEW", mesh_name=mesh_name, new_name=name
    ) == {"FINISHED"}
    stored = stored_name(instance.edit_target.mesh_index_lookup[0].data.shape_keys, f"{mesh_name}__{name}")
    bpy.ops.ed.undo_push(message="After RBF shape")
    if not bpy.ops.ed.undo.poll():
        pytest.skip("Blender build has no headless undo context")
    assert bpy.ops.ed.undo() == {"FINISHED"}
    instance = get_active_rig_instance()
    instance.edit_target.initialize(update_rbf_solver_list=False)
    assert stored not in shapes.blocks(instance.edit_target.mesh_index_lookup[0])
    assert not any(
        item.name == name
        for solver in instance.rbf_editor.rbf_solver_list
        for pose in solver.poses
        for item in pose.driven
    )
    RBFEditor.for_instance(instance).revert(bpy.context)


def test_apply_shape_value_and_mirror_membership(character):
    from character_dna.editors.rbf_editor import preview
    from character_dna.editors.rbf_editor.properties import (
        DEFAULT_BONE_MIRROR_REGEX,
        DEFAULT_POSE_MIRROR_REGEX,
        DEFAULT_SOLVER_MIRROR_REGEX,
    )

    instance = character
    editor, _ = enter(instance)
    mesh_name = instance.edit_target.dna_reader.getMeshName(0)
    obj = instance.edit_target.mesh_index_lookup[0]
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="NEW", mesh_name=mesh_name, new_name="test_L"
    ) == {"FINISHED"}
    pose = utilities.get_active_pose(instance)
    pose.scale_factor = 0.5
    preview.show(instance, pose)
    key = shapes.blocks(obj)[f"{mesh_name}__test_L"]
    key.value = 0.25
    assert bpy.ops.character_dna.apply_rbf_pose_edits() == {"FINISHED"}
    assert pose.driven[-1].scalar_value == pytest.approx(0.5, abs=1e-5)
    assert bpy.ops.character_dna.apply_rbf_pose_edits() == {"FINISHED"}
    assert pose.driven[-1].scalar_value == pytest.approx(0.5, abs=1e-5)
    obj.shape_key_add(name=f"{mesh_name}__test_R", from_mix=False).value = 0.0
    success, message, _ = utilities.add_rbf_pose(instance, "custom_l_pose", from_pose=pose)
    assert success, message
    success, message, _ = utilities.mirror_pose(
        instance, DEFAULT_SOLVER_MIRROR_REGEX, DEFAULT_BONE_MIRROR_REGEX, DEFAULT_POSE_MIRROR_REGEX
    )
    assert success, message
    solver = utilities.get_active_solver(instance)
    assert solver.name == "calf_r_UERBFSolver"
    assert utilities.get_active_pose(instance).driven[-1].name == "test_R"
    assert utilities.get_active_pose(instance).driven[-1].scalar_value == pytest.approx(0.5, abs=1e-5)
    assert all(any(item.name == "test_R" for item in pose.driven) for pose in solver.poses)
    editor.revert(bpy.context)


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
@pytest.mark.parametrize("mode", ["SCENE", "DNA"])
def test_merged_shape_commit_and_split(character, mode, tmp_path):
    from character_dna.dna_io import document, get_dna_reader
    from character_dna.editors.converter import scene_merge
    from character_dna.shape_names import stored_name

    instance = character
    if mode == "DNA":
        instance.converter_merge_mode = "DNA"
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    registry = scene_merge.surfaces(instance)
    identities = {lod: (obj.as_pointer(), obj.data.as_pointer()) for lod, obj in registry.items()}
    editor, _ = enter(instance)
    mesh_name = instance.edit_target.dna_reader.getMeshName(0)
    name = "merged_corrective__" + "long_expression_" * 5
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="NEW", mesh_name=mesh_name, new_name=name
    ) == {"FINISHED"}
    key_name = stored_name(instance.edit_target.mesh_index_lookup[0].data.shape_keys, f"{mesh_name}__{name}")
    for proxy in registry.values():
        if proxy["dna_merge_body"] == instance.edit_target.mesh_index_lookup[0]:
            assert key_name in shapes.blocks(proxy)
    editor.commit(bpy.context)
    assert {
        lod: (obj.as_pointer(), obj.data.as_pointer()) for lod, obj in scene_merge.surfaces(instance).items()
    } == identities
    if mode == "DNA":
        from pathlib import Path

        path = Path(instance["combined_dna_path"])
        assert path.is_file()
        outputs = document.split_files(path, tmp_path / "rbf_split")
        reader = get_dna_reader(outputs["body"])
        try:
            assert name in [reader.getBlendShapeChannelName(i) for i in range(reader.getBlendShapeChannelCount())]
            assert any(
                item.data_type == "SHAPE_KEY" and item.name == name
                for solver in core.get_rbf_solver_data(reader)
                for pose in solver.poses
                for item in pose.driven
            )
        finally:
            release_dna_handle(reader)
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    assert key_name in shapes.blocks(instance.edit_target.mesh_index_lookup[0])
