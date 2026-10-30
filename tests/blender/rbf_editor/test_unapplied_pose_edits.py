"""Commit warns about live pose edits without changing the scene before confirmation."""

from pathlib import Path
from types import SimpleNamespace

import bpy
import pytest

from rbf_editor.test_shape_driven import enter
from test_merge_split import character as character_fixture

from character_dna.editors.rbf_editor import preview, shapes, utilities
from character_dna.editors.rbf_editor.core import get_rbf_solver_data
from character_dna.editors.rbf_editor.operators import CommitRBFSolverChanges
from fixtures.rigs import RigProfile


paired_character = character_fixture


@pytest.fixture
def character(rig_factory, tmp_path, compact_dna):
    with rig_factory.lease(RigProfile(components=("body",), asset=str(compact_dna)), tmp_path) as instance:
        yield instance


@pytest.fixture
def component_character(rig_factory, tmp_path, kind, compact_dna):
    with rig_factory.lease(RigProfile(components=(kind,), asset=str(compact_dna)), tmp_path) as instance:
        yield instance


@pytest.mark.parametrize("kind", ["head", "body"])
def test_detection_matches_apply_and_ignores_unchanged_preview(component_character, kind):
    instance = component_character
    editor, solver = enter(instance, kind)
    for index in range(len(solver.poses)):
        solver.poses_active_index = index
        assert not utilities.active_pose_has_unapplied_edits(instance), solver.poses[index].name
    solver.poses_active_index = 1
    pose = utilities.get_active_pose(instance)
    driven = next(item for item in pose.driven if item.data_type == "BONE")
    bone = instance.edit_target.rig.pose.bones[driven.name]
    bone.location.x += 1e-5
    assert not utilities.active_pose_has_unapplied_edits(instance)
    bone.location.x -= 1e-5
    members = {item.name for item in (*pose.driven, *pose.drivers)}
    unrelated = next(b for b in instance.edit_target.rig.pose.bones if b.name not in members)
    original = unrelated.matrix_basis.copy()
    unrelated.location.x += 0.1
    assert not utilities.active_pose_has_unapplied_edits(instance)
    unrelated.matrix_basis = original
    for field in ("location", "rotation_euler", "scale"):
        getattr(bone, field)[0] += 0.03
        assert utilities.active_pose_has_unapplied_edits(instance), field
        assert bpy.ops.character_dna.apply_rbf_pose_edits() == {"FINISHED"}
        assert not utilities.active_pose_has_unapplied_edits(instance), field
    driver = instance.edit_target.driver_rig.pose.bones[pose.drivers[0].name]
    driver.rotation_mode = "XYZ"
    driver.rotation_euler[0] += 0.02
    assert utilities.active_pose_has_unapplied_edits(instance)
    assert bpy.ops.character_dna.apply_rbf_pose_edits() == {"FINISHED"}
    assert not utilities.active_pose_has_unapplied_edits(instance)
    editor.revert(bpy.context)


def test_dialog_is_read_only_and_names_pose(character):
    instance = character
    editor, _solver = enter(instance)
    pose = utilities.get_active_pose(instance)
    bone = instance.edit_target.rig.pose.bones[pose.driven[0].name]
    bone.location.x += 0.025
    original = Path(instance.body_dna_file_path).read_bytes()
    stored = tuple(pose.driven[0].location)
    displayed = bone.matrix_basis.copy()
    calls, messages = [], []
    operation = SimpleNamespace(
        _instance=lambda: instance,
        _pose_identity=CommitRBFSolverChanges._pose_identity,
        _report_error=lambda message: pytest.fail(message),
        execute=lambda _context: pytest.fail("Dialog committed before confirmation"),
        layout=SimpleNamespace(label=lambda *, text, **_kwargs: messages.append(text), prop=lambda *_args: None),
    )
    context = SimpleNamespace(
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda *_args, **kwargs: calls.append(kwargs) or {"RUNNING_MODAL"}
        )
    )
    assert CommitRBFSolverChanges.invoke(operation, context, None) == {"RUNNING_MODAL"}
    CommitRBFSolverChanges.draw(operation, context)
    assert calls[0]["title"] == "Unapplied Pose Edits"
    assert any(pose.name in text for text in messages)
    assert operation.apply_pose_edits
    assert tuple(pose.driven[0].location) == stored
    assert bone.matrix_basis == displayed
    assert Path(instance.body_dna_file_path).read_bytes() == original
    # Closing/cancelling the dialog has no execute step and leaves the edit session intact.
    assert editor.is_editing
    utilities.get_active_solver(instance).poses_active_index = 2
    operation._report_error = lambda message: messages.append(message) or {"CANCELLED"}
    assert CommitRBFSolverChanges.execute(operation, bpy.context) == {"CANCELLED"}
    assert "The active pose changed. Open Commit again." in messages
    assert Path(instance.body_dna_file_path).read_bytes() == original
    editor.revert(bpy.context)


@pytest.mark.parametrize("apply", [True, False])
def test_commit_respects_explicit_pose_choice(character, apply):
    instance = character
    _editor, solver = enter(instance)
    pose = utilities.get_active_pose(instance)
    solver_name, pose_name, bone_name = solver.name, pose.name, pose.driven[0].name
    driven = pose.driven[0]
    bone = instance.edit_target.rig.pose.bones[bone_name]
    original = tuple(driven.location)
    bone.location.x += 0.025
    expected = tuple(utilities.driven_bone_channels(instance, pose, driven, bone)[0]) if apply else original
    before = Path(instance.body_dna_file_path).read_bytes()
    with pytest.raises(RuntimeError, match="unapplied edits"):
        bpy.ops.character_dna.commit_rbf_solver_changes()
    assert Path(instance.body_dna_file_path).read_bytes() == before
    assert instance.rbf_editor.is_editing
    assert bpy.ops.character_dna.commit_rbf_solver_changes(apply_pose_edits=apply) == {"FINISHED"}
    saved_solver = next(s for s in get_rbf_solver_data(instance.body_dna_reader) if s.name == solver_name)
    saved_pose = next(p for p in saved_solver.poses if p.name == pose_name)
    saved = next(d for d in saved_pose.driven if d.name == bone_name)
    assert tuple(saved.location) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("mode", ["separate", "scene", "combined"])
@pytest.mark.rig_profile(compact=True)
@pytest.mark.rig_profile(compact=True, pose_bones=())
def test_shape_value_commit_applies_scaled_pose_coefficient(paired_character, mode):
    instance = paired_character
    if mode != "separate":
        instance.converter_merge_mode = "DNA" if mode == "combined" else "SCENE"
        assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    _editor, solver = enter(instance)
    mesh = instance.edit_target.dna_reader.getMeshName(0)
    assert bpy.ops.character_dna.add_rbf_driven(
        data_type="SHAPE_KEY", shape_source="NEW", mesh_name=mesh, new_name="commit_corrective"
    ) == {"FINISHED"}
    pose = utilities.get_active_pose(instance)
    pose.scale_factor = 0.5
    preview.show(instance, pose)
    assert not utilities.active_pose_has_unapplied_edits(instance)
    # Editing the editor's coefficient slider is already stored, so needs no warning.
    pose.driven[-1].scalar_value = 0.8
    assert not utilities.active_pose_has_unapplied_edits(instance)
    shapes.channel_blocks(instance, "commit_corrective")[0][1].value = 0.25
    assert utilities.active_pose_has_unapplied_edits(instance)
    solver_name, pose_name = solver.name, pose.name
    assert bpy.ops.character_dna.commit_rbf_solver_changes(apply_pose_edits=True) == {"FINISHED"}
    saved_solver = next(s for s in get_rbf_solver_data(instance.body_dna_reader) if s.name == solver_name)
    saved_pose = next(p for p in saved_solver.poses if p.name == pose_name)
    saved = next(d for d in saved_pose.driven if d.name == "commit_corrective")
    assert saved.scalar_value == pytest.approx(0.5, abs=1e-5)
