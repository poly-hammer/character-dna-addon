"""Neutral generation must preserve DNA behavior and provide recoverable bulk writes."""

import json
import queue

from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from test_dna_operation_backups import configure_backups, entries
from test_merge_split import character as character_fixture, coordinates

from character_dna import shape_names
from character_dna.bindings import riglogic
from character_dna.dna_io import release_dna_handle
from character_dna.editors.backup_manager import core as backups
from character_dna.editors.converter import scene_merge
from character_dna.editors.rbf_editor import generation
from character_dna.editors.shape_key_editor.operators import (
    GenerateNeutralShapeKeys,
    GenerateRBFShapeKeys,
    ImportShapeKeys,
)
from character_dna.runtime import engine


character = character_fixture


def test_head_generate_is_neutral_and_import_still_loads_deltas(character):
    instance = character
    path = Path(instance.head_dna_file_path)
    original = path.read_bytes()
    for operator in (GenerateNeutralShapeKeys, ImportShapeKeys):
        operation = SimpleNamespace(
            _timer=None, _import_instance=None, _native_was_active=False, neutral=operator.neutral
        )
        component = ImportShapeKeys.get_component(operation)
        commands = queue.Queue()
        operator.set_commands_queue(operation, bpy.context, component, commands)
        try:
            # Exercise the actual operator queue on one DNA target, including Basis setup.
            for _ in range(2):
                index, mesh, _, kwargs, callback = commands.get()
                callback(**kwargs(index, mesh))
        finally:
            operator.finish(operation, bpy.context)
        keys = instance.head_mesh.data.shape_keys
        delta = coordinates(keys.key_blocks[1].data) - coordinates(keys.reference_key.data)
        if operator.neutral:
            np.testing.assert_array_equal(delta, 0)
        else:
            assert np.max(np.abs(delta)) > 1e-6
        assert path.read_bytes() == original


def sample_pose(reader, solver, position):
    manager = riglogic.RigLogic(reader, riglogic.Configuration(), None)
    state = riglogic.RigInstance(rigLogic=manager, memRes=None)
    indices = list(reader.getRBFSolverRawControlIndices(solver))
    values = list(reader.getRBFSolverRawControlValues(solver))
    try:
        for index, value in zip(indices, values[position * len(indices) : (position + 1) * len(indices)], strict=True):
            state.setRawControl(index, value)
        manager.calculate(state)
        return np.array(state.getJointOutputs()), np.array(state.getBlendShapeOutputs())
    finally:
        release_dna_handle(state)
        release_dna_handle(manager)


def assert_body_keys(instance, names):
    obj = instance.component("body").mesh
    actual = (
        {
            shape_names.logical_name(obj.data.shape_keys, key.name).split("__", 1)[-1]
            for key in obj.data.shape_keys.key_blocks[1:]
        }
        if obj.data.shape_keys
        else set()
    )
    assert actual == names
    if obj.data.shape_keys:
        basis = coordinates(obj.data.shape_keys.reference_key.data)
        for key in obj.data.shape_keys.key_blocks[1:]:
            np.testing.assert_array_equal(coordinates(key.data), basis)


@pytest.mark.parametrize("mode", ["SEPARATE", "SCENE", "DNA"])
def test_body_generate_routes_keys_once_and_restores_pre_post_backups(character, tmp_path, monkeypatch, mode):
    instance = character
    configure_backups(monkeypatch, tmp_path)
    if mode != "SEPARATE":
        instance.converter_merge_mode = mode
        assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    instance.editor_component = "body"
    planned = generation.plan(instance)
    names = {entry["name"] for entry in planned["entries"]}
    assert len(names) > 50 and "default" not in names
    head_before = Path(instance.head_dna_file_path).read_bytes()
    reader = instance.body_dna_reader
    before = {position: sample_pose(reader, 0, position)[0] for position in (0, 1, 2)}
    proxy = scene_merge.proxy_for(instance)
    assert bpy.ops.character_dna.generate_rbf_shape_keys() == {"FINISHED"}
    assert_body_keys(instance, names)
    reader = instance.body_dna_reader
    channels = {reader.getBlendShapeChannelName(i): i for i in range(reader.getBlendShapeChannelCount())}
    for position in (0, 1, 2):
        joints, weights = sample_pose(reader, 0, position)
        np.testing.assert_allclose(joints, before[position], atol=1e-6)
        for entry in (item for item in planned["entries"] if item["solver"] == 0):
            pose = reader.getRBFSolverPoseIndices(0)[position]
            assert weights[channels[entry["name"]]] == pytest.approx(float(pose == entry["pose"]), abs=1e-5)
    assert Path(instance.head_dna_file_path).read_bytes() == head_before
    assert not generation.plan(instance)["entries"]
    current = Path(instance.body_dna_file_path).read_bytes()
    assert generation.generate(instance) == 0
    assert Path(instance.body_dna_file_path).read_bytes() == current
    saved = entries(instance)
    pre, before_metadata = saved[backups.BackupType.PRE_DNA_GENERATE_RBF_SHAPES.value]
    post, after_metadata = saved[backups.BackupType.POST_DNA_GENERATE_RBF_SHAPES.value]
    assert before_metadata["operation_id"] == after_metadata["operation_id"]
    # Restore from the opposite component tab: generation history always owns body keys.
    instance.editor_component = "head"
    assert backups.restore_backup(instance, pre)
    assert instance.body_dna_reader.getBlendShapeChannelCount() == 0
    assert_body_keys(instance, set())
    assert backups.restore_backup(instance, post)
    assert_body_keys(instance, names)
    assert scene_merge.proxy_for(instance) == proxy
    assert not engine.binding_issues(instance)


def test_generation_confirmation_is_read_only_and_reports_exact_count(character):
    instance = character
    instance.editor_component = "body"
    original = Path(instance.body_dna_file_path).read_bytes()
    invoked = []
    operation = SimpleNamespace(report=lambda *_args: None)
    context = SimpleNamespace(
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda *_args, **kwargs: invoked.append(kwargs) or {"RUNNING_MODAL"}
        )
    )
    assert GenerateRBFShapeKeys.invoke(operation, context, None) == {"RUNNING_MODAL"}
    count = len(json.loads(operation.confirmed_plan)["entries"])
    messages = []
    operation.layout = SimpleNamespace(label=lambda *, text, **_kwargs: messages.append(text))
    GenerateRBFShapeKeys.draw(operation, context)
    assert f"This will add {count} shape keys corresponding to each RBF pose." in messages
    assert invoked[0]["confirm_text"] == "Generate"
    assert Path(instance.body_dna_file_path).read_bytes() == original
    assert_body_keys(instance, set())


@pytest.mark.parametrize("phase", ["write", "post_backup"])
def test_body_generation_failure_restores_files_and_keys(character, tmp_path, monkeypatch, phase):
    instance = character
    instance.editor_component = "body"
    configure_backups(monkeypatch, tmp_path)
    path = Path(instance.body_dna_file_path)
    original = path.read_bytes()
    if phase == "write":
        write = generation.shape_channels.write

        def failed(*args):
            write(*args)
            raise ValueError("Simulated write failure")

        monkeypatch.setattr(generation.shape_channels, "write", failed)
    else:
        create = backups.create_backup
        monkeypatch.setattr(
            backups,
            "create_backup",
            lambda instance, kind, **kwargs: None
            if kind == backups.BackupType.POST_DNA_GENERATE_RBF_SHAPES
            else create(instance, kind, **kwargs),
        )
    with pytest.raises((ValueError, RuntimeError)):
        generation.generate(instance)
    assert path.read_bytes() == original
    assert_body_keys(instance, set())
    assert not engine.binding_issues(instance)
