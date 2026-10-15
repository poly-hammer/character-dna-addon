"""Component routing and preservation of real head/body RBF behavior."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from character_dna.bindings import dna
from character_dna.dna_io import get_dna_reader, release_dna_handle
from character_dna.dna_io.coordinates import configuration
from character_dna.editors.rbf_editor.core import commit_rbf_data_to_dna, get_rbf_driven_data, get_rbf_solver_data


RESOURCES = Path(__file__).parent / "test_files/dna/default"


@pytest.fixture(params=["head", "body"])
def component_reader(request):
    path = next(RESOURCES.rglob(f"{request.param}.dna"))
    reader = get_dna_reader(path)
    yield request.param, reader
    release_dna_handle(reader)


def solver_data(reader):
    return [{**asdict(solver), "__property_group__": "RBFSolverData"} for solver in get_rbf_solver_data(reader)]


def roundtrip(reader, data):
    stream = dna.MemoryStream()
    writer = dna.BinaryStreamWriter(stream)
    commit_rbf_data_to_dna(reader, writer, data)
    stream.seek(0)
    result = dna.BinaryStreamReader(stream, configuration(), None)
    result.read()
    assert dna.Status.isOk(), dna.Status.get().message
    return stream, writer, result


def test_rbf_noop_preserves_behavior(component_reader):
    _, reader = component_reader
    stream, writer, result = roundtrip(reader, solver_data(reader))
    try:
        assert result.getRBFSolverCount() == reader.getRBFSolverCount()
        assert result.getRBFPoseCount() == reader.getRBFPoseCount()
        for index in range(reader.getRBFPoseCount()):
            for field in ("InputControlIndices", "OutputControlIndices", "OutputControlWeights"):
                np.testing.assert_allclose(
                    getattr(result, f"getRBFPose{field}")(index),
                    getattr(reader, f"getRBFPose{field}")(index),
                    atol=1e-6,
                )
        for index in range(reader.getJointGroupCount()):
            for field in ("InputIndices", "OutputIndices", "Values", "LODs", "JointIndices"):
                np.testing.assert_allclose(
                    getattr(result, f"getJointGroup{field}")(index),
                    getattr(reader, f"getJointGroup{field}")(index),
                    atol=2e-5,
                )
        for index in range(reader.getRBFSolverCount()):
            np.testing.assert_allclose(
                result.getRBFSolverRawControlValues(index), reader.getRBFSolverRawControlValues(index), atol=1e-6
            )
    finally:
        for handle in (result, writer, stream):
            release_dna_handle(handle)


def test_removing_solver_preserves_surviving_pose_routes(component_reader):
    _, reader = component_reader
    data = solver_data(reader)
    if len(data) < 2:
        pytest.skip("Single-solver DNA")
    data.pop(0)
    expected = {
        pose["name"]: (
            list(reader.getRBFPoseInputControlIndices(pose["pose_index"])),
            list(reader.getRBFPoseOutputControlIndices(pose["pose_index"])),
        )
        for solver in data
        for pose in solver["poses"]
        if pose["name"] != "default"
    }
    stream, writer, result = roundtrip(reader, data)
    try:
        assert result.getRBFSolverCount() == reader.getRBFSolverCount() - 1
        for index in range(result.getRBFPoseCount()):
            name = result.getRBFPoseName(index)
            if name in expected:
                assert list(result.getRBFPoseInputControlIndices(index)) == expected[name][0]
                assert list(result.getRBFPoseOutputControlIndices(index)) == expected[name][1]
    finally:
        for handle in (result, writer, stream):
            release_dna_handle(handle)


def test_joint_edit_preserves_other_pose_contributions(component_reader):
    _, reader = component_reader
    data = solver_data(reader)
    pose = next(
        pose
        for solver in data
        for pose in solver["poses"]
        if pose["name"] != "default" and any(item["data_type"] == "BONE" for item in pose["driven"])
    )
    driven = next(item for item in pose["driven"] if item["data_type"] == "BONE")
    driven["location"][0] += 0.001
    stream, writer, result = roundtrip(reader, data)
    try:
        after = next(
            item
            for item in get_rbf_driven_data(result, pose["pose_index"])
            if item.joint_index == driven["joint_index"]
        )
        assert after.location[0] == pytest.approx(driven["location"][0], abs=1e-6)
        for index in range(reader.getRBFPoseCount()):
            if index == pose["pose_index"]:
                continue
            before = get_rbf_driven_data(reader, index)
            after = get_rbf_driven_data(result, index)
            for old, new in zip(before, after, strict=True):
                np.testing.assert_allclose(new.location, old.location, atol=1e-6)
                np.testing.assert_allclose(new.euler_rotation, old.euler_rotation, atol=1e-6)
    finally:
        for handle in (result, writer, stream):
            release_dna_handle(handle)


def test_head_shared_pose_can_add_a_neutral_driven_joint():
    from character_dna.bindings import riglogic
    from character_dna.editors.rbf_editor.core import initialize_rbf_driven_data

    reader = get_dna_reader(RESOURCES / "head.dna")
    data = solver_data(reader)
    pose = next(pose for pose in data[0]["poses"] if pose["name"] != "default" and pose["driven"])
    existing = {item["joint_index"] for item in pose["driven"] if item["data_type"] == "BONE"}
    joint = next(index for index in range(reader.getJointCount()) if index not in existing)
    item = asdict(initialize_rbf_driven_data(reader.getJointName(joint), pose["pose_index"], -1))
    item["joint_index"] = joint
    pose["driven"].append(item)
    stream, writer, result = roundtrip(reader, data)
    manager = state = None
    try:
        restored = get_rbf_driven_data(result, pose["pose_index"])
        assert joint in {item.joint_index for item in restored}
        manager = riglogic.RigLogic(result, riglogic.Configuration(), None)
        state = riglogic.RigInstance(rigLogic=manager, memRes=None)
        manager.calculate(state)
        assert np.isfinite(state.getJointOutputs()).all()
    finally:
        for handle in (state, manager, result, writer, stream, reader):
            release_dna_handle(handle)


def test_component_selection_routes_and_locks_sessions(tmp_path):
    import hashlib
    import shutil

    import bpy

    from character_dna.editors.rbf_editor.editor import RBFEditor
    from character_dna.runtime import engine
    from character_dna.utilities import get_active_rig_instance
    from constants import TEST_DNA_FOLDER

    bpy.ops.wm.read_homefile(use_empty=True)
    for name in ("head.dna", "body.dna", "ExportManifest.json"):
        shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, tmp_path / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / "head.dna"),
        include_body=True,
        import_face_board=False,
        import_mesh=False,
        import_bones=True,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    )
    instance = get_active_rig_instance()
    try:
        for kind, other in (("body", "head"), ("head", "body")):
            instance.editor_component = kind
            reader = instance.edit_target.dna_reader
            assert len(instance.rbf_editor.rbf_solver_list) == reader.getRBFSolverCount()
            assert instance.rbf_editor.rbf_solver_list[0].name == reader.getRBFSolverName(0)
            digest = hashlib.sha256((tmp_path / f"{other}.dna").read_bytes()).digest()
            editor = RBFEditor.for_instance(instance)
            editor.enter(bpy.context)
            instance.editor_component = other
            assert instance.editor_component == kind
            assert editor.component.value == kind
            editor.commit(bpy.context)
            assert hashlib.sha256((tmp_path / f"{other}.dna").read_bytes()).digest() == digest
            assert not editor.is_editing
            assert engine.binding_issues(instance) == []
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


def test_body_shape_channels_bind_and_preview(tmp_path):
    import queue

    from types import SimpleNamespace

    import bpy

    from character_dna.dna_io import get_dna_writer
    from character_dna.editors.shape_key_editor.utilities import build_shape_key_import_commands
    from character_dna.runtime import controller, engine
    from character_dna.utilities import get_active_body, get_active_rig_instance

    reader = get_dna_reader(RESOURCES / "body.dna")
    writer = get_dna_writer(tmp_path / "body.dna")
    writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    writer.setBlendShapeChannelName(0, "test_corrective")
    writer.setBlendShapeChannelIndices(0, [0])
    for lod in range(reader.getLODCount()):
        writer.setLODBlendShapeChannelMapping(lod, 0)
    writer.setBlendShapeChannelLODs([1] * reader.getLODCount())
    writer.setBlendShapeChannelInputIndices([0])
    writer.setBlendShapeChannelOutputIndices([0])
    writer.setBlendShapeChannelIndex(0, 0, 0)
    writer.setBlendShapeTargetDeltas(0, 0, [[1.0, 0.0, 0.0]])
    writer.setBlendShapeTargetVertexIndices(0, 0, [0])
    writer.write()
    assert dna.Status.isOk(), dna.Status.get().message
    release_dna_handle(writer)
    release_dna_handle(reader)
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / "body.dna"),
        include_body=False,
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    )
    instance = get_active_rig_instance()
    try:
        assert instance.editor_component == "body"
        assert bpy.ops.character_dna.import_shape_keys.poll()
        was_active = controller.suspend(instance)
        assert not bpy.ops.character_dna.import_shape_keys.poll()
        commands = queue.Queue()
        build_shape_key_import_commands(get_active_body(), commands)
        while not commands.empty():
            index, mesh, _, kwargs, operation = commands.get_nowait()
            operation(**kwargs(index, mesh))
        from character_dna.editors.shape_key_editor.operators import ImportShapeKeys

        operator = SimpleNamespace(
            _timer=None, _commands_queue=commands, _import_instance=instance, _native_was_active=was_active
        )
        ImportShapeKeys.finish(operator, bpy.context)
        assert bpy.ops.character_dna.import_shape_keys.poll()
        assert len(instance.shape_key_editor.shape_key_list) == 1
        assert engine.binding_issues(instance) == []
        block = instance.edit_target.shape_key_blocks[0][0]
        assert block.id_data.animation_data.drivers.find(block.path_from_id("value")) is not None
        assert engine.preview_shape(instance, 0, 0.65, "body")
        bpy.context.view_layer.update()
        evaluated = block.id_data.evaluated_get(bpy.context.view_layer.depsgraph)
        assert evaluated.key_blocks[block.name].value == pytest.approx(0.65)
        from character_dna.editors.raw_control_editor.editor import RawControlEditor
        from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor

        instance.raw_control_editor.raw_controls_active_index = instance.raw_control_editor.raw_controls.find("default")
        for editor_type in (ShapeKeyEditor, RawControlEditor):
            editor = editor_type.for_instance(instance)
            editor.enter(bpy.context)
            assert editor.component.value == "body"
            editor.revert(bpy.context)
            assert engine.binding_issues(instance) == []
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)
