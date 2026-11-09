"""Pose menus must not resolve a scene RNA path once per pose button."""

from types import SimpleNamespace

import bpy
import pytest

from character_dna.ui import callbacks, view_3d


@pytest.fixture
def pose_browser_scenes():
    previous = bpy.context.window.scene
    scenes = [bpy.data.scenes.new("PoseBrowserA"), bpy.data.scenes.new("PoseBrowserB")]
    bpy.context.window.scene = scenes[0]
    try:
        yield scenes
    finally:
        bpy.context.window.scene = previous
        for scene in scenes:
            bpy.data.scenes.remove(scene)


def test_pose_browser_forwards_selection_and_tracks_active_scene(pose_browser_scenes, tmp_path):
    first, second = pose_browser_scenes
    browser = bpy.context.window_manager.character_dna
    poses = callbacks.ensure_face_pose_metadata()
    first_pose, second_pose = poses[:2]
    first.character_dna.face_board.face_pose_previews = first_pose["id"]
    second.character_dna.face_board.face_pose_previews = second_pose["id"]

    assert browser.face_pose_previews == first_pose["id"]
    bpy.context.window.scene = second
    assert browser.face_pose_previews == second_pose["id"]
    browser.face_pose_previews = first_pose["id"]
    assert second.character_dna.face_board.face_pose_previews == first_pose["id"]
    bpy.context.window.scene = first
    browser.face_pose_previews = second_pose["id"]
    assert first.character_dna.face_board.face_pose_previews == second_pose["id"]
    assert "face_pose_previews" not in browser  # No duplicated selection state.

    path = tmp_path / "pose_browser.blend"
    bpy.data.libraries.write(str(path), {first})
    with bpy.data.libraries.load(str(path), link=False) as (source, target):
        target.scenes = source.scenes
    loaded = target.scenes[0]
    try:
        bpy.context.window.scene = loaded
        assert browser.face_pose_previews == second_pose["id"]
    finally:
        bpy.context.window.scene = first
        bpy.data.scenes.remove(loaded)


def test_pose_browser_uses_filtered_cache_and_stable_values(pose_browser_scenes):
    browser = bpy.context.window_manager.character_dna
    face_board = bpy.context.scene.character_dna.face_board
    metadata = callbacks.ensure_face_pose_metadata()
    face_board.category = "visemes"
    items = callbacks.get_face_pose_ui_items(browser, bpy.context)
    assert items is callbacks.get_face_pose_previews_items(face_board, bpy.context)
    assert items is callbacks.get_face_pose_ui_items(browser, bpy.context)
    expected = {entry["id"]: entry["value"] for entry in metadata if entry["category"] == "visemes"}
    assert {item[0]: item[4] for item in items} == expected
    browser.face_pose_previews = items[-1][0]
    assert face_board.get("face_pose_previews") == items[-1][4]
    face_board.category = "ALL"
    assert browser.face_pose_previews == items[-1][0]
    assert callbacks.get_face_pose_ui_items(browser, None) == ()


def test_pose_browser_empty_filter_and_readonly_selection(pose_browser_scenes, monkeypatch):
    from character_dna.properties import face_pose_preview_collections

    browser = bpy.context.window_manager.character_dna
    face_board = bpy.context.scene.character_dna.face_board
    pose = callbacks.ensure_face_pose_metadata()[1]
    face_board.face_pose_previews = pose["id"]
    monkeypatch.setattr(callbacks, "is_face_board_readonly", lambda _instance: True)
    callbacks.set_face_pose_ui_value(browser, 0)
    assert face_board.face_pose_previews == pose["id"]
    monkeypatch.setattr(callbacks, "is_face_board_readonly", lambda _instance: False)
    monkeypatch.setattr(
        callbacks, "get_face_pose_previews_items", lambda *_: [(callbacks.NO_FACE_POSE, "No Poses", "", 0, 0)]
    )
    monkeypatch.setattr(face_pose_preview_collections["face_poses"], "face_pose_preview_values", {0})
    assert callbacks.get_face_pose_ui_value(browser) == 0
    callbacks.set_face_pose_ui_value(browser, 0)
    assert face_board.get("face_pose_previews") == pose["value"]


def test_pose_panel_draws_both_pickers_from_window_manager(pose_browser_scenes, monkeypatch):
    drawn = []

    class Layout:
        def row(self, **_kwargs):
            return self

        column = row

        def prop(self, owner, name, **_kwargs):
            drawn.append((owner, name))

        template_icon_view = prop

        def label(self, *_args, **_kwargs):
            pass

        popover = operator = label

    monkeypatch.setattr(view_3d, "get_active_rig_instance", lambda: None)
    monkeypatch.setattr(view_3d, "valid_rig_instance_exists", lambda *_: "")
    view_3d.CHARACTER_DNA_PT_face_board.draw(SimpleNamespace(layout=Layout()), bpy.context)
    pickers = [owner for owner, name in drawn if name == "face_pose_previews"]
    assert len(pickers) == 2
    assert all(owner.id_data == bpy.context.window_manager for owner in pickers)
    assert all(owner.id_data == bpy.context.scene for owner, name in drawn if name != "face_pose_previews")


def test_pose_browser_applies_the_same_native_pose_as_scene_api(rig_factory, compact_dna, tmp_path):
    from fixtures.compact_dna import STATE_BONES
    from fixtures.rigs import RigProfile

    profile = RigProfile(components=("head",), asset=str(compact_dna), pose_bones=STATE_BONES, face_board=True)
    with rig_factory.lease(profile, tmp_path) as instance:
        face_board = bpy.context.scene.character_dna.face_board
        browser = bpy.context.window_manager.character_dna
        poses = {pose["name"]: pose["id"] for pose in callbacks.ensure_face_pose_metadata()}
        face_board.face_pose_previews = poses["Jaw Open"]
        bpy.context.view_layer.update()
        expected = instance.head_rig.pose.bones["FACIAL_C_Jaw"].matrix_basis.copy()
        face_board.face_pose_previews = poses["Neutral"]
        bpy.context.view_layer.update()
        assert instance.head_rig.pose.bones["FACIAL_C_Jaw"].matrix_basis != expected
        browser.face_pose_previews = poses["Jaw Open"]
        bpy.context.view_layer.update()
        assert face_board.face_pose_previews == poses["Jaw Open"]
        actual = instance.head_rig.pose.bones["FACIAL_C_Jaw"].matrix_basis
        for row, reference in zip(actual, expected, strict=True):
            assert tuple(row) == pytest.approx(tuple(reference), abs=1e-6)
