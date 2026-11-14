"""Component selection must reset the owning rig and reflect available DNA shapes."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import bpy
import numpy as np
import pytest

from character_dna.constants import ToolInfo
from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigProfile


@pytest.fixture
def character(tmp_path, request, rig_factory, compact_dna):
    profile = RigProfile(
        components=("head", "body") if getattr(request, "param", True) else ("head",),
        asset=str(compact_dna),
        pose_bones=STATE_BONES,
    )
    with rig_factory.lease(profile, tmp_path) as instance:
        instance.editor_component = "head"
        yield instance


@pytest.mark.parametrize("character", [True, False], indirect=True, ids=["constrained", "standalone"])
def test_head_default_pose_resets_drivers_and_preview(character):
    from character_dna.editors.rbf_editor.editor import RBFEditor

    instance = character
    target = instance.edit_target
    solver = instance.rbf_editor.rbf_solver_list[0]
    default_index = solver.poses.find("default")
    turned_index = solver.poses.find("turnUp_55")
    assert default_index >= 0 and turned_index >= 0
    driver_names = [driver.name for driver in solver.poses[default_index].drivers]
    assert driver_names
    editor = RBFEditor.for_instance(instance)

    def displayed_neck():
        bpy.context.view_layer.update()
        rig = target.rig.evaluated_get(bpy.context.view_layer.depsgraph)
        return np.array([rig.pose.bones[name].matrix for name in driver_names])

    for editing in (False, True):
        if editing:
            editor.enter(bpy.context)
        for reset in (True, False):
            instance.rbf_editor.reset_pose_on_change = reset
            solver.poses_active_index = default_index
            baseline = displayed_neck()
            unrelated = None
            if target.driver_rig != target.rig:
                unrelated = target.driver_rig.pose.bones["calf_l"]
                unrelated.rotation_mode = "XYZ"
                unrelated.rotation_euler.x = 0.3
                unrelated_basis = np.array(unrelated.matrix_basis)
            solver.poses_active_index = turned_index
            assert not np.allclose(displayed_neck(), baseline, atol=1e-4)
            solver.poses_active_index = default_index
            np.testing.assert_allclose(displayed_neck(), baseline, atol=2e-5)
            for name in driver_names:
                np.testing.assert_allclose(target.driver_rig.pose.bones[name].matrix_basis, np.eye(4), atol=2e-5)
            if unrelated is not None:
                np.testing.assert_allclose(unrelated.matrix_basis, unrelated_basis, atol=2e-5)
            if editing:
                preview = instance.data[target.cache_key("rbf_preview")]
                baseline_data = instance.data[target.cache_key("rbf_preview_baseline")]
                assert baseline_data["pose"] == solver.poses[default_index].as_pointer()
                np.testing.assert_allclose(preview.state.getBlendShapeOutputs(), 0, atol=2e-5)
        if editing:
            editor.revert(bpy.context)


def test_solver_list_refresh_preserves_live_pose(character):
    instance = character
    solver = instance.rbf_editor.rbf_solver_list[0]
    solver.poses_active_index = solver.poses.find("default")
    pose_index = solver.poses_active_index
    bone = instance.edit_target.driver_rig.pose.bones[solver.poses[pose_index].drivers[0].name]
    bone.rotation_mode = "XYZ"
    bone.rotation_euler.x = 0.2
    before = np.array(bone.matrix_basis)

    instance.update_body_rbf_solver_list()

    assert instance.rbf_editor.rbf_solver_list[0].poses_active_index == pose_index
    np.testing.assert_allclose(bone.matrix_basis, before, atol=2e-5)


def test_component_switch_refreshes_both_solver_editors(character, monkeypatch):
    from character_dna.editors.rbf_editor import ui, utilities
    from character_dna.editors.swing_twist_editor import core
    from character_dna.editors.swing_twist_editor.ui import CHARACTER_DNA_PT_swing_twist_editor

    def draw_list(_layout, context, **kwargs):
        for key in ("list_path", "active_index_path"):
            context.scene.path_resolve(kwargs[key].removeprefix("scene."))

    monkeypatch.setattr(ui, "draw_ui_list", create_autospec(ui.draw_ui_list, side_effect=draw_list))
    for kind in ("body", "head", "body", "head"):
        character.editor_component = kind
        properties = character.rbf_editor
        assert len(properties.rbf_solver_list) == character.edit_target.dna_reader.getRBFSolverCount()
        assert utilities.get_active_solver(character) is not None
        assert utilities.get_active_pose(character) is not None
        assert len(character.swing_twist_editor.setups) == len(core.read_setups(character.edit_target.dna_reader))
        panel = SimpleNamespace(layout=MagicMock())
        ui.CHARACTER_DNA_PT_rbf_editor.draw(panel, bpy.context)
        CHARACTER_DNA_PT_swing_twist_editor.draw(panel, bpy.context)
        assert not ui.RbfEditorSubPanelBase.poll(bpy.context)
        assert not ui.CHARACTER_DNA_PT_rbf_editor_driven_sub_panel.poll(bpy.context)
        if kind == "body":
            assert len(properties.rbf_solver_list) > 3
            properties.rbf_solver_list_active_index = 3
            character.swing_twist_editor.active_index = len(character.swing_twist_editor.setups) - 1


@pytest.mark.parametrize("index", [-1, 3])
def test_rbf_panels_and_selection_helpers_handle_stale_indices(minimal_character, tmp_path, index):
    from character_dna.editors.rbf_editor import ui, utilities

    instance = minimal_character
    dna_path = tmp_path / "head.dna"
    dna_path.touch()
    instance["head_dna_file_path"] = str(dna_path)
    instance.data[instance.edit_target.cache_key("initialized")] = True
    properties = instance.rbf_editor
    solver = properties.rbf_solver_list.add()
    pose = solver.poses.add()
    pose.driven.add()
    properties.is_editing = True
    panel = SimpleNamespace(layout=MagicMock())

    properties.rbf_solver_list_active_index = index
    assert utilities.get_active_solver(instance) is None
    assert utilities.get_active_pose(instance) is None
    assert utilities.get_active_driven(instance) is None
    assert not ui.RbfEditorSubPanelBase.poll(bpy.context)
    assert not ui.CHARACTER_DNA_PT_rbf_editor_driven_sub_panel.poll(bpy.context)
    for panel_type in (
        ui.CHARACTER_DNA_PT_rbf_editor,
        ui.CHARACTER_DNA_PT_rbf_editor_solver_settings_sub_panel,
        ui.CHARACTER_DNA_PT_rbf_editor_poses_sub_panel,
        ui.CHARACTER_DNA_PT_rbf_editor_drivers_sub_panel,
        ui.CHARACTER_DNA_PT_rbf_editor_driven_sub_panel,
    ):
        panel_type.draw(panel, bpy.context)

    properties.rbf_solver_list_active_index = 0
    solver["poses_active_index"] = index
    assert utilities.get_active_pose(instance) is None
    assert utilities.get_active_driven(instance) is None
    assert not ui.CHARACTER_DNA_PT_rbf_editor_driven_sub_panel.poll(bpy.context)
    ui.CHARACTER_DNA_PT_rbf_editor_poses_sub_panel.draw(panel, bpy.context)
    ui.CHARACTER_DNA_PT_rbf_editor_drivers_sub_panel.draw(panel, bpy.context)
    ui.CHARACTER_DNA_PT_rbf_editor_driven_sub_panel.draw(panel, bpy.context)

    solver["poses_active_index"] = 0
    pose["driven_active_index"] = index
    assert utilities.get_active_driven(instance) is None
    properties.is_editing = False


def test_rbf_refresh_recovers_stale_and_empty_selection(character, monkeypatch):
    from character_dna.editors.rbf_editor import core, utilities

    properties = character.rbf_editor
    for solver_index, pose_index in ((3, 0), (-1, 0), (0, 999), (0, -1)):
        properties.rbf_solver_list_active_index = solver_index
        properties.rbf_solver_list[0]["poses_active_index"] = pose_index
        character.update_body_rbf_solver_list()
        assert utilities.get_active_solver(character) is not None
        assert utilities.get_active_pose(character) is not None

    with monkeypatch.context() as patch:
        patch.setattr(core, "get_rbf_solver_data", create_autospec(core.get_rbf_solver_data, return_value=[]))
        character.update_body_rbf_solver_list()
    assert not properties.rbf_solver_list
    assert properties.rbf_solver_list_active_index == -1
    assert utilities.get_active_solver(character) is None
    character.update_body_rbf_solver_list()
    assert properties.rbf_solver_list_active_index == 0
    assert utilities.get_active_pose(character) is not None


class EmptyStateLayout:
    def __init__(self):
        self.labels = []
        self.operators = []
        self.properties = []

    def row(self, **_kwargs):
        return self

    def label(self, *, text, **_kwargs):
        self.labels.append(text)

    def prop(self, _owner, name, **_kwargs):
        self.properties.append(name)

    def operator(self, name, **_kwargs):
        self.operators.append(name)


def test_shape_import_availability_matches_selected_dna(character):
    from character_dna.editors.shape_key_editor.operators import ImportShapeKeys
    from character_dna.editors.shape_key_editor.ui import CHARACTER_DNA_PT_shape_keys

    instance = character
    getattr(bpy.context.window_manager, ToolInfo.NAME).progress = 1
    for kind, count in (("head", 782), ("body", 0), ("head", 782)):
        instance.editor_component = kind
        assert instance.edit_target.dna_reader.getBlendShapeChannelCount() == count
        assert not instance.shape_key_editor.shape_key_list
        layout = EmptyStateLayout()
        panel = SimpleNamespace(layout=layout)
        panel._draw_empty_state = MethodType(CHARACTER_DNA_PT_shape_keys._draw_empty_state, panel)
        CHARACTER_DNA_PT_shape_keys.draw(panel, bpy.context)
        assert f"No shape keys on {instance.name}. {count} in {kind} DNA" in layout.labels
        assert (f"{ToolInfo.NAME}.import_shape_keys" in layout.operators) == bool(count)
        assert "generate_neutral_shapes" not in layout.properties
        generate = "generate_rbf_shape_keys" if kind == "body" else "generate_neutral_shape_keys"
        assert f"{ToolInfo.NAME}.{generate}" in layout.operators
        assert layout.scale_y == 1.5
        assert bpy.ops.character_dna.import_shape_keys.poll() == bool(count)
        reports = []
        operator = SimpleNamespace(report=lambda _level, message, reports=reports: reports.append(message))
        assert ImportShapeKeys.validate(operator, bpy.context) == bool(count)
        assert bool(reports) == (count == 0)
