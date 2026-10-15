"""Component selection must reset the owning rig and reflect available DNA shapes."""

import shutil

from types import MethodType, SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.constants import ToolInfo
from character_dna.utilities import get_active_rig_instance
from constants import TEST_DNA_FOLDER


@pytest.fixture
def character(tmp_path, request):
    bpy.ops.wm.read_homefile(use_empty=True)
    for name in ("head.dna", "body.dna", "ExportManifest.json"):
        shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, tmp_path / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / "head.dna"),
        include_body=getattr(request, "param", True),
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    )
    instance = get_active_rig_instance()
    instance.editor_component = "head"
    try:
        yield instance
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


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
        assert ("generate_neutral_shapes" in layout.properties) == bool(count)
        assert bpy.ops.character_dna.import_shape_keys.poll() == bool(count)
        reports = []
        operator = SimpleNamespace(report=lambda _level, message, reports=reports: reports.append(message))
        assert ImportShapeKeys.validate(operator, bpy.context) == bool(count)
        assert bool(reports) == (count == 0)
