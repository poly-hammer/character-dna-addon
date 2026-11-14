"""The viewer must work after authoring caches are released by load or undo."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.editors.behavior_viewer import html_report, operators, ui, utilities
from character_dna.runtime import controller, engine
from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigProfile


@pytest.fixture
def character(rig_factory, compact_dna, tmp_path):
    profile = RigProfile(asset=str(compact_dna), pose_bones=STATE_BONES)
    with rig_factory.lease(profile, tmp_path) as instance:
        yield instance


def _cold_cache(instance):
    with controller.preserve_bindings():
        instance.destroy()
    assert instance.head_dna_reader is None
    assert instance.body_dna_reader is None
    assert engine.active(instance)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_cold_viewer_load_preserves_native_pose(character, kind):
    instance = character
    instance.editor_component = kind
    target = instance.edit_target
    if kind == "head":
        reader = target.dna_reader
        jaw = next(j for j in range(reader.getRawControlCount()) if reader.getRawControlName(j).endswith(".jawOpen"))
        target.state.setRawControl(jaw, 0.6)
        target.manager.calculate(target.state)
        assert engine.preview_controls(instance, target.state, kind)
    bpy.context.view_layer.update()
    before_controls = list(target.state.getRawControlValues())
    before_pose = np.asarray([bone.matrix for bone in target.rig.pose.bones])
    carriers = [(obj.as_pointer(), obj["token"]) for obj in engine.carriers(instance)]
    _cold_cache(instance)

    assert ui.CHARACTER_DNA_PT_behavior_viewer.poll(bpy.context)
    assert bpy.ops.character_dna.behavior_viewer_load.poll()
    assert bpy.ops.character_dna.behavior_viewer_show_graph.poll()
    # Polling must not initialize data during a read-only UI draw.
    assert target.dna_reader is None
    assert bpy.ops.character_dna.behavior_viewer_load() == {"FINISHED"}

    assert target.initialized
    assert list(target.state.getRawControlValues()) == pytest.approx(before_controls, abs=1e-6)
    assert [(obj.as_pointer(), obj["token"]) for obj in engine.carriers(instance)] == carriers
    bpy.context.view_layer.update()
    np.testing.assert_allclose(np.asarray([bone.matrix for bone in target.rig.pose.bones]), before_pose, atol=1e-6)
    assert engine.active(instance) and not engine.binding_issues(instance)
    assert instance.behavior_viewer.activate_on_click
    if kind == "head":
        assert instance.behavior_viewer.target_pose


def test_graph_command_loads_cold_data_without_opening_a_browser(character, monkeypatch, tmp_path):
    instance = character
    instance.editor_component = "head"
    _cold_cache(instance)
    captured = {}

    def render(graph, title, **kwargs):
        captured["graph"] = graph
        return tmp_path / "behavior.html"

    monkeypatch.setattr(html_report, "render_graph_html", render)
    monkeypatch.setattr(operators, "_open_report", lambda *_args: {"FINISHED"})
    assert bpy.ops.character_dna.behavior_viewer_show_graph() == {"FINISHED"}
    assert instance.head_dna_reader is not None
    assert captured["graph"].nodes
    assert engine.active(instance)


def test_unavailable_and_hidden_pro_viewer(monkeypatch, tmp_path):
    target = SimpleNamespace(rig=object(), dna_file_path=str(tmp_path / "missing.dna"))
    instance = SimpleNamespace(edit_target=target)
    monkeypatch.setattr(ui, "get_active_rig_instance", lambda: instance)
    monkeypatch.setattr(ui, "pro_features_visible", lambda: True)
    assert not ui.CHARACTER_DNA_PT_behavior_viewer.poll(bpy.context)
    assert not utilities.behavior_available(None)
    target.dna_file_path = str(tmp_path / "head.dna")
    (tmp_path / "head.dna").touch()
    assert ui.CHARACTER_DNA_PT_behavior_viewer.poll(bpy.context)
    monkeypatch.setattr(ui, "pro_features_visible", lambda: False)
    assert not ui.CHARACTER_DNA_PT_behavior_viewer.poll(bpy.context)
